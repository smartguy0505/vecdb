//! Compact storage for full vectors, used to re-rank candidates exactly
//! (or nearly exactly) after a compressed search.
//!
//! | codec | bytes per dim | error |
//! |---|---:|---|
//! | `f32` | 4 | none |
//! | `f16` | 2 | ~3 significant digits; values beyond +/-65504 are rejected |
//! | `bf16` | 2 | ~2-3 significant digits, full f32 range |
//! | `sq8` | 1 | per-dimension affine 8-bit (min..max split into 255 steps) |

use crate::error::{Error, Result};
use crate::store::check_dim;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Codec {
    F32,
    F16,
    BF16,
    SQ8,
}

impl Codec {
    pub const ALL: [Codec; 4] = [Codec::F32, Codec::F16, Codec::BF16, Codec::SQ8];

    pub fn name(self) -> &'static str {
        match self {
            Codec::F32 => "f32",
            Codec::F16 => "f16",
            Codec::BF16 => "bf16",
            Codec::SQ8 => "sq8",
        }
    }

    pub fn parse(s: &str) -> Option<Self> {
        Codec::ALL.into_iter().find(|c| c.name() == s.to_ascii_lowercase())
    }

    pub fn bytes_per_dim(self) -> usize {
        match self {
            Codec::F32 => 4,
            Codec::F16 | Codec::BF16 => 2,
            Codec::SQ8 => 1,
        }
    }
}

// ---------- f16 (IEEE 754 binary16) ----------

/// Rounds to the nearest f16 (ties to even). Overflow gives +/-inf, NaN stays NaN.
pub fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xFF) as i32;
    let man = b & 0x7F_FFFF;
    if exp == 0xFF {
        return sign | 0x7C00 | if man != 0 { 0x200 } else { 0 };
    }
    let e = exp - 127 + 15;
    if e >= 0x1F {
        return sign | 0x7C00;
    }
    if e <= 0 {
        // f16 subnormal (or zero): value / 2^-24, rounded.
        if e < -10 {
            return sign;
        }
        let m = man | 0x80_0000;
        let shift = (14 - e) as u32; // 14..=24
        let mut h = m >> shift;
        let rem = m & ((1 << shift) - 1);
        let half = 1 << (shift - 1);
        if rem > half || (rem == half && h & 1 == 1) {
            h += 1;
        }
        return sign | h as u16;
    }
    let mut h = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1FFF;
    if rem > 0x1000 || (rem == 0x1000 && h & 1 == 1) {
        h += 1; // may carry into the exponent; reaching 0x7C00 correctly means inf
    }
    sign | h as u16
}

pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1F) as u32;
    let man = (h & 0x3FF) as u32;
    let bits = match exp {
        0 if man == 0 => sign,
        0 => {
            // Subnormal: normalize the mantissa.
            let shift = man.leading_zeros() - 21; // bring the top set bit to bit 10
            let m = (man << shift) & 0x3FF;
            sign | ((113 - shift) << 23) | (m << 13)
        }
        0x1F => sign | 0x7F80_0000 | (man << 13),
        _ => sign | ((exp + 112) << 23) | (man << 13),
    };
    f32::from_bits(bits)
}

// ---------- bf16 (top 16 bits of f32) ----------

/// Rounds to the nearest bf16 (ties to even). NaN stays NaN.
pub fn f32_to_bf16(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x40) as u16;
    }
    let rounded = b.wrapping_add(0x7FFF + ((b >> 16) & 1));
    (rounded >> 16) as u16
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

// ---------- encoded vector storage ----------

#[derive(Debug, Clone)]
enum Storage {
    F32(Vec<f32>),
    Half(Vec<u16>),
    Byte { codes: Vec<u8>, min: Vec<f32>, step: Vec<f32> },
}

/// Row-major vectors stored with one [`Codec`].
#[derive(Debug, Clone)]
pub struct EncodedVectors {
    codec: Codec,
    dim: usize,
    len: usize,
    storage: Storage,
}

impl EncodedVectors {
    /// Encodes `data` (row-major, `dim` per row). `sq8` learns each
    /// dimension's min and max from `data`.
    pub fn encode(codec: Codec, dim: usize, data: &[f32]) -> Result<Self> {
        check_dim(dim)?;
        if data.len() % dim != 0 {
            return Err(Error::InvalidArgument(format!(
                "{} values do not form rows of dimension {dim}",
                data.len()
            )));
        }
        let len = data.len() / dim;
        let storage = match codec {
            Codec::F32 => Storage::F32(data.to_vec()),
            Codec::F16 | Codec::BF16 => {
                let (to, from): (fn(f32) -> u16, fn(u16) -> f32) = if codec == Codec::F16 {
                    (f32_to_f16, f16_to_f32)
                } else {
                    (f32_to_bf16, bf16_to_f32)
                };
                let mut out = Vec::with_capacity(data.len());
                for &x in data {
                    let h = to(x);
                    if x.is_finite() && !from(h).is_finite() {
                        return Err(Error::InvalidArgument(format!(
                            "value {x} is outside the {} range",
                            codec.name()
                        )));
                    }
                    out.push(h);
                }
                Storage::Half(out)
            }
            Codec::SQ8 => {
                let mut min = vec![f32::INFINITY; dim];
                let mut max = vec![f32::NEG_INFINITY; dim];
                for row in data.chunks_exact(dim) {
                    for d in 0..dim {
                        min[d] = min[d].min(row[d]);
                        max[d] = max[d].max(row[d]);
                    }
                }
                if len == 0 {
                    min.fill(0.0);
                    max.fill(0.0);
                }
                let step: Vec<f32> = min.iter().zip(&max).map(|(&lo, &hi)| (hi - lo) / 255.0).collect();
                let mut codes = Vec::with_capacity(data.len());
                for row in data.chunks_exact(dim) {
                    for d in 0..dim {
                        codes.push(sq8_code(row[d], min[d], step[d]));
                    }
                }
                Storage::Byte { codes, min, step }
            }
        };
        Ok(Self { codec, dim, len, storage })
    }

    pub fn codec(&self) -> Codec {
        self.codec
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn bytes_per_vector(&self) -> usize {
        self.dim * self.codec.bytes_per_dim()
    }

    /// Row `row` as f32: borrowed directly for `f32`, otherwise decoded into
    /// `buf` (which must hold `dim` values). Panics if `row >= len()`.
    pub fn get<'a>(&'a self, row: usize, buf: &'a mut [f32]) -> &'a [f32] {
        let (lo, hi) = (row * self.dim, (row + 1) * self.dim);
        match &self.storage {
            Storage::F32(v) => &v[lo..hi],
            Storage::Half(v) => {
                let from = if self.codec == Codec::F16 { f16_to_f32 } else { bf16_to_f32 };
                for (o, &h) in buf[..self.dim].iter_mut().zip(&v[lo..hi]) {
                    *o = from(h);
                }
                &buf[..self.dim]
            }
            Storage::Byte { codes, min, step } => {
                for (d, (o, &c)) in buf[..self.dim].iter_mut().zip(&codes[lo..hi]).enumerate() {
                    *o = min[d] + c as f32 * step[d];
                }
                &buf[..self.dim]
            }
        }
    }
}

#[inline]
fn sq8_code(x: f32, min: f32, step: f32) -> u8 {
    if step > 0.0 { ((x - min) / step).round().clamp(0.0, 255.0) as u8 } else { 0 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::SplitMix64;

    #[test]
    fn f16_known_values() {
        for (x, h) in [
            (0.0f32, 0x0000u16),
            (-0.0, 0x8000),
            (1.0, 0x3C00),
            (-2.0, 0xC000),
            (0.5, 0x3800),
            (65504.0, 0x7BFF),
            (6.103_515_6e-5, 0x0400),   // smallest normal
            (5.960_464_5e-8, 0x0001),   // smallest subnormal
            (65520.0, 0x7C00),          // rounds up to inf
            (f32::INFINITY, 0x7C00),
            (1e-9, 0x0000),             // underflows to zero
        ] {
            assert_eq!(f32_to_f16(x), h, "{x}");
        }
        assert!(f16_to_f32(f32_to_f16(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_round_trips_every_value() {
        for h in 0..=u16::MAX {
            let x = f16_to_f32(h);
            if x.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16(x), h, "{h:#06x} -> {x}");
        }
    }

    #[test]
    fn f16_rounds_to_nearest() {
        let mut rng = SplitMix64::new(5);
        for _ in 0..200_000 {
            // Spread over normal and subnormal f16 ranges.
            let x = rng.uniform(-1.0, 1.0) * 2f32.powi(rng.below(40) as i32 - 25);
            let h = f32_to_f16(x);
            let err = (f16_to_f32(h) - x).abs();
            for nb in [h.wrapping_sub(1), h.wrapping_add(1)] {
                let y = f16_to_f32(nb);
                if y.is_finite() && (y.is_sign_negative() == x.is_sign_negative() || y == 0.0) {
                    assert!(err <= (y - x).abs(), "{x}: chose {h:#06x}, neighbor {nb:#06x} is closer");
                }
            }
        }
    }

    #[test]
    fn bf16_rounding() {
        assert_eq!(f32_to_bf16(1.0), 0x3F80);
        assert!(bf16_to_f32(f32_to_bf16(3.0e38)).is_finite());
        assert!(bf16_to_f32(f32_to_bf16(f32::MAX)).is_infinite()); // rounds past the max
        // 1 + 2^-8 is exactly halfway between bf16 1.0 and 1 + 2^-7: ties to even (1.0).
        assert_eq!(f32_to_bf16(1.0 + 2f32.powi(-8)), 0x3F80);
        assert_eq!(f32_to_bf16(1.0 + 3.0 * 2f32.powi(-8)), 0x3F82);
        assert!(bf16_to_f32(f32_to_bf16(f32::NAN)).is_nan());
        let mut rng = SplitMix64::new(6);
        for _ in 0..10_000 {
            let x = rng.uniform(-1000.0, 1000.0);
            let y = bf16_to_f32(f32_to_bf16(x));
            assert!((y - x).abs() <= x.abs() * 2f32.powi(-8) + 1e-30, "{x} -> {y}");
        }
    }

    #[test]
    fn encoded_vectors_error_bounds() {
        let mut rng = SplitMix64::new(9);
        let dim = 37;
        let data: Vec<f32> = (0..dim * 50).map(|_| rng.uniform(-3.0, 5.0)).collect();
        let mut buf = vec![0.0; dim];
        for codec in Codec::ALL {
            let enc = EncodedVectors::encode(codec, dim, &data).unwrap();
            assert_eq!((enc.len(), enc.dim()), (50, dim));
            assert_eq!(enc.bytes_per_vector(), dim * codec.bytes_per_dim());
            let tol = match codec {
                Codec::F32 => 0.0,
                Codec::F16 => 5.0 * 2f32.powi(-11),
                Codec::BF16 => 5.0 * 2f32.powi(-8),
                Codec::SQ8 => 8.0 / 255.0 / 2.0 + 1e-6, // half a step of the widest range
            };
            for row in 0..50 {
                let got = enc.get(row, &mut buf).to_vec();
                for (g, x) in got.iter().zip(&data[row * dim..(row + 1) * dim]) {
                    assert!((g - x).abs() <= tol, "{} row {row}: {x} -> {g}", codec.name());
                }
            }
        }
    }

    #[test]
    fn f16_rejects_out_of_range_and_sq8_handles_constant_dims() {
        assert!(EncodedVectors::encode(Codec::F16, 2, &[1.0, 70000.0]).is_err());
        assert!(EncodedVectors::encode(Codec::BF16, 2, &[1.0, 70000.0]).is_ok());
        let enc = EncodedVectors::encode(Codec::SQ8, 2, &[7.0, 1.0, 7.0, 2.0]).unwrap();
        let mut buf = [0.0; 2];
        assert_eq!(enc.get(0, &mut buf), &[7.0, 1.0]);
        assert_eq!(enc.get(1, &mut buf), &[7.0, 2.0]);
        assert!(EncodedVectors::encode(Codec::SQ8, 3, &[]).unwrap().is_empty());
    }
}
