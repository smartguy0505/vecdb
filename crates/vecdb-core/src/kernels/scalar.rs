//! Portable reference kernels: fixed-width chunks and independent
//! accumulators, which LLVM auto-vectorizes (SSE2 on default x86_64 builds;
//! NEON expected on aarch64, not yet tested there).

use super::check_batch;

/// Independent accumulators: 16 f32 = two AVX2 registers or four SSE/NEON
/// registers, enough to hide add latency.
const LANES: usize = 16;

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..LANES {
            acc[i] += x[i] * y[i];
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ar.iter().zip(br) {
        tail += x * y;
    }
    reduce(&acc) + tail
}

#[inline]
pub fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_squared: length mismatch");
    let (ac, ar) = a.as_chunks::<LANES>();
    let (bc, br) = b.as_chunks::<LANES>();
    let mut acc = [0.0f32; LANES];
    for (x, y) in ac.iter().zip(bc) {
        for i in 0..LANES {
            let d = x[i] - y[i];
            acc[i] += d * d;
        }
    }
    let mut tail = 0.0f32;
    for (x, y) in ar.iter().zip(br) {
        let d = x - y;
        tail += d * d;
    }
    reduce(&acc) + tail
}

#[inline]
fn reduce(acc: &[f32; LANES]) -> f32 {
    // Pairwise sum: a little more accurate than a left fold.
    let mut v = *acc;
    let mut width = LANES;
    while width > 1 {
        width /= 2;
        for i in 0..width {
            v[i] += v[i + width];
        }
    }
    v[0]
}

pub fn dot_batch(query: &[f32], rows: &[f32], out: &mut [f32]) {
    check_batch(query, rows, out);
    for (row, o) in rows.chunks_exact(query.len()).zip(out.iter_mut()) {
        *o = dot(query, row);
    }
}

pub fn l2_squared_batch(query: &[f32], rows: &[f32], out: &mut [f32]) {
    check_batch(query, rows, out);
    for (row, o) in rows.chunks_exact(query.len()).zip(out.iter_mut()) {
        *o = l2_squared(query, row);
    }
}
