//! Little-endian encoding for the collection's own files (manifest, WAL,
//! tombstones), and their atomic replacement on disk.

use std::fs::{self, File};
use std::io::Write;
use std::path::Path;

use crate::error::{Error, Result};
use crate::format::{sync_parent_dir, tmp_path};

#[derive(Default)]
pub(crate) struct Enc {
    pub buf: Vec<u8>,
}

impl Enc {
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }

    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }

    pub fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }

    pub fn f32s(&mut self, v: &[f32]) {
        self.buf.reserve(v.len() * 4);
        v.iter().for_each(|x| self.buf.extend_from_slice(&x.to_le_bytes()));
    }

    /// A string with a u16 length.
    pub fn str(&mut self, s: &str) {
        self.u16(s.len() as u16);
        self.buf.extend_from_slice(s.as_bytes());
    }

    /// Appends a CRC-32C of everything written so far.
    pub fn crc(&mut self) {
        let c = crate::crc32c::crc32c(&self.buf);
        self.u32(c);
    }
}

pub(crate) struct Dec<'a> {
    buf: &'a [u8],
    pos: usize,
    what: &'static str,
}

impl<'a> Dec<'a> {
    pub fn new(buf: &'a [u8], what: &'static str) -> Self {
        Self { buf, pos: 0, what }
    }

    /// Checks the CRC-32C in the last 4 bytes and decodes the rest.
    pub fn checked(buf: &'a [u8], what: &'static str) -> Result<Self> {
        if buf.len() < 4 {
            return Err(Error::Corrupt(format!("{what}: too short")));
        }
        let (body, tail) = buf.split_at(buf.len() - 4);
        if crate::crc32c::crc32c(body) != u32::from_le_bytes(tail.try_into().unwrap()) {
            return Err(Error::Corrupt(format!("{what}: checksum mismatch")));
        }
        Ok(Self::new(body, what))
    }

    pub fn corrupt(&self, m: &str) -> Error {
        Error::Corrupt(format!("{}: {m}", self.what))
    }

    pub fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.buf.len() - self.pos < n {
            return Err(self.corrupt("truncated"));
        }
        let s = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(s)
    }

    pub fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    pub fn u16(&mut self) -> Result<u16> {
        Ok(u16::from_le_bytes(self.take(2)?.try_into().unwrap()))
    }

    pub fn u32(&mut self) -> Result<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }

    pub fn u64(&mut self) -> Result<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn i64(&mut self) -> Result<i64> {
        Ok(i64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }

    pub fn f32(&mut self) -> Result<f32> {
        Ok(f32::from_bits(self.u32()?))
    }

    pub fn f32s(&mut self, out: &mut [f32]) -> Result<()> {
        let b = self.take(out.len() * 4)?;
        for (x, c) in out.iter_mut().zip(b.as_chunks::<4>().0) {
            *x = f32::from_le_bytes(*c);
        }
        Ok(())
    }

    pub fn str(&mut self) -> Result<&'a str> {
        let n = self.u16()? as usize;
        std::str::from_utf8(self.take(n)?).map_err(|_| self.corrupt("string is not UTF-8"))
    }

    pub fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Fails unless every byte was decoded.
    pub fn done(&self) -> Result<()> {
        if self.remaining() != 0 {
            return Err(self.corrupt("trailing bytes"));
        }
        Ok(())
    }
}

/// Writes `bytes` to `path` through `<path>.tmp`, fsynced and renamed, so a
/// crash leaves the old file or the new one. `sync_dir` also fsyncs the
/// directory (callers writing several files sync it once at the end).
pub(crate) fn write_file(path: &Path, bytes: &[u8], sync_dir: bool) -> Result<()> {
    let tmp = tmp_path(path);
    let result = (|| {
        let mut f = File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
        fs::rename(&tmp, path)?;
        if sync_dir {
            sync_parent_dir(path)?;
        }
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&tmp);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_and_checks() {
        let mut e = Enc::default();
        e.u8(7);
        e.u16(300);
        e.u32(70_000);
        e.u64(1 << 40);
        e.i64(-5);
        e.f32(-0.5);
        e.f32s(&[1.0, 2.0]);
        e.str("héllo");
        e.crc();
        let mut d = Dec::checked(&e.buf, "test").unwrap();
        assert_eq!((d.u8().unwrap(), d.u16().unwrap(), d.u32().unwrap()), (7, 300, 70_000));
        assert_eq!((d.u64().unwrap(), d.i64().unwrap(), d.f32().unwrap()), (1 << 40, -5, -0.5));
        let mut v = [0.0; 2];
        d.f32s(&mut v).unwrap();
        assert_eq!(v, [1.0, 2.0]);
        assert_eq!(d.str().unwrap(), "héllo");
        d.done().unwrap();
        assert!(d.u8().is_err());

        let mut bad = e.buf.clone();
        bad[3] ^= 1;
        assert!(matches!(Dec::checked(&bad, "test"), Err(Error::Corrupt(_))));
        assert!(Dec::checked(&[1, 2], "test").is_err());
    }
}
