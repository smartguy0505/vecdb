//! CRC-32C (Castagnoli), used for on-disk checksums.
//!
//! Chosen over IEEE CRC-32 because x86 (SSE4.2) and ARMv8 compute it in hardware.
//! x86_64 uses the SSE4.2 instruction when the CPU has it (about 8 bytes per
//! 3 cycles, which matters because the SSD index checks every record it
//! reads); other CPUs use the table below. Both give identical values.

const POLY: u32 = 0x82F6_3B78; // reflected Castagnoli polynomial

const TABLE: [u32; 256] = make_table();

const fn make_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 { POLY ^ (c >> 1) } else { c >> 1 };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

/// Streaming CRC-32C hasher.
#[derive(Clone, Debug)]
pub struct Crc32c {
    state: u32,
}

impl Default for Crc32c {
    fn default() -> Self {
        Self::new()
    }
}

impl Crc32c {
    pub fn new() -> Self {
        Self { state: !0 }
    }

    pub fn update(&mut self, data: &[u8]) {
        self.state = match crate::kernels::crc32c_update(self.state, data) {
            Some(c) => c,
            None => update_table(self.state, data),
        };
    }

    pub fn finalize(&self) -> u32 {
        !self.state
    }
}

fn update_table(mut c: u32, data: &[u8]) -> u32 {
    for &b in data {
        c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c
}

pub fn crc32c(data: &[u8]) -> u32 {
    let mut h = Crc32c::new();
    h.update(data);
    h.finalize()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_check_value() {
        // Standard CRC-32C check value.
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
    }

    #[test]
    fn hardware_matches_table() {
        let data: Vec<u8> = (0..4099u32).map(|i| (i.wrapping_mul(2_654_435_761) >> 13) as u8).collect();
        for len in [0usize, 1, 7, 8, 9, 63, 64, 776, 4096, 4099] {
            for start in [0usize, 1, 3] {
                if start + len > data.len() {
                    continue;
                }
                let part = &data[start..start + len];
                let table = !update_table(!0, part);
                assert_eq!(crc32c(part), table, "len {len} start {start}");
            }
        }
    }

    #[test]
    fn streaming_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut h = Crc32c::new();
        for part in data.chunks(37) {
            h.update(part);
        }
        assert_eq!(h.finalize(), crc32c(&data));
    }
}
