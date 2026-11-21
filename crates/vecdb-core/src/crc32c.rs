//! CRC-32C (Castagnoli), used for on-disk checksums.
//!
//! Chosen over IEEE CRC-32 because x86 (SSE4.2) and ARMv8 compute it in hardware;
//! a hardware path can replace this table version later without a format change.

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
        let mut c = self.state;
        for &b in data {
            c = TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
        }
        self.state = c;
    }

    pub fn finalize(&self) -> u32 {
        !self.state
    }
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
    fn streaming_matches_one_shot() {
        let data: Vec<u8> = (0..1000u32).map(|i| (i * 7 % 251) as u8).collect();
        let mut h = Crc32c::new();
        for part in data.chunks(37) {
            h.update(part);
        }
        assert_eq!(h.finalize(), crc32c(&data));
    }
}
