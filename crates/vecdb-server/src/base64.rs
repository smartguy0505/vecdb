//! Base64, the standard alphabet.

const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";

/// The value of each character; 255 for the others.
const VALUES: [u8; 256] = {
    let mut v = [255; 256];
    let mut i = 0;
    while i < 64 {
        v[ALPHABET[i] as usize] = i as u8;
        i += 1;
    }
    v
};

/// Decodes `s`, whether or not it is padded; `None` if it is not base64.
pub fn decode(s: &str) -> Option<Vec<u8>> {
    let b = s.as_bytes();
    let end = b.iter().rposition(|&c| c != b'=').map_or(0, |i| i + 1);
    let (body, padding) = b.split_at(end);
    if padding.len() > 2 || (!padding.is_empty() && b.len() % 4 != 0) || body.len() % 4 == 1 {
        return None;
    }
    let mut out = Vec::with_capacity(body.len() / 4 * 3 + 2);
    for chunk in body.chunks(4) {
        let mut x = 0u32;
        for (i, &c) in chunk.iter().enumerate() {
            match VALUES[c as usize] {
                255 => return None,
                v => x |= (v as u32) << (18 - 6 * i),
            }
        }
        let n = chunk.len() - 1;
        // The bits beyond the last byte say nothing, and have to be zero.
        if x & ((1 << (24 - 8 * n)) - 1) != 0 {
            return None;
        }
        out.extend_from_slice(&x.to_be_bytes()[1..1 + n]);
    }
    Some(out)
}

#[cfg(test)]
pub fn encode(b: &[u8]) -> String {
    let mut out = String::new();
    for chunk in b.chunks(3) {
        let mut x = [0; 4];
        x[1..1 + chunk.len()].copy_from_slice(chunk);
        let x = u32::from_be_bytes(x);
        for i in 0..4 {
            let c = ALPHABET[(x >> (18 - 6 * i)) as usize & 63] as char;
            out.push(if i <= chunk.len() { c } else { '=' });
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_refuses() {
        // RFC 4648.
        for (plain, coded) in
            [("", ""), ("f", "Zg=="), ("fo", "Zm8="), ("foo", "Zm9v"), ("foob", "Zm9vYg=="), ("foobar", "Zm9vYmFy")]
        {
            assert_eq!(decode(coded).unwrap(), plain.as_bytes());
            assert_eq!(decode(coded.trim_end_matches('=')).unwrap(), plain.as_bytes());
            assert_eq!(encode(plain.as_bytes()), coded);
        }
        let all: Vec<u8> = (0..=255).chain(0..7).collect();
        for n in 0..all.len() {
            assert_eq!(decode(&encode(&all[..n])).unwrap(), &all[..n]);
        }
        for bad in ["Z", "Zg=", "Zg===", "=", "====", "Zm9v=", "Zm9vZ", "Zm 9", "Zm-9", "Zm9v\n", "Zh==", "Zm9=", "é"] {
            assert_eq!(decode(bad), None, "{bad:?}");
        }
    }
}
