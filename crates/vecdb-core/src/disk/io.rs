//! Block reads for the SSD index.
//!
//! - On Linux the index file is opened with `O_DIRECT`, so reads bypass the
//!   page cache. Latency is then the SSD's, and a large index does not evict
//!   the rest of the machine's cache. Where the filesystem refuses `O_DIRECT`
//!   (tmpfs, for example) the file is opened normally and
//!   [`DataFile::is_direct`] reports it.
//! - A [`Reader`] issues a batch of reads and returns once all of them have
//!   finished: through io_uring on Linux (one submission, every read in
//!   flight at once), or with positioned reads one after another (the
//!   portable fallback).
//!
//! The reader owns its buffers, and they start on 4096-byte boundaries as
//! `O_DIRECT` requires. This is the only module of the SSD index with
//! `unsafe` code: io_uring writes into a buffer after the call that queued
//! the read has returned, which the compiler cannot check.

#![allow(unsafe_code)]

use std::fs::File;
use std::io;
use std::path::Path;

/// Buffer and offset alignment of every read.
pub(crate) const ALIGN: usize = 4096;
/// Most reads one batch may hold (the io_uring queue size).
pub(crate) const DEPTH: usize = 64;

/// How search reads blocks from the SSD.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum IoMode {
    /// io_uring where the kernel allows it, positioned reads otherwise.
    Auto,
    /// io_uring only; creating a search context fails where it is unavailable.
    Uring,
    /// Positioned reads, one after another (no read concurrency in a search).
    Pread,
}

impl IoMode {
    pub fn parse(s: &str) -> Option<Self> {
        match s.to_ascii_lowercase().as_str() {
            "auto" => Some(IoMode::Auto),
            "uring" | "io_uring" => Some(IoMode::Uring),
            "pread" => Some(IoMode::Pread),
            _ => None,
        }
    }
}

/// The index file, opened for block reads.
#[derive(Debug)]
pub(crate) struct DataFile {
    file: File,
    direct: bool,
}

impl DataFile {
    pub fn open(path: &Path, want_direct: bool) -> io::Result<Self> {
        #[cfg(target_os = "linux")]
        if want_direct {
            use std::os::unix::fs::OpenOptionsExt;
            match std::fs::OpenOptions::new().read(true).custom_flags(libc::O_DIRECT).open(path) {
                Ok(file) => return Ok(Self { file, direct: true }),
                // The filesystem does not support O_DIRECT: fall back below.
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {}
                Err(e) => return Err(e),
            }
        }
        let _ = want_direct;
        Ok(Self { file: File::open(path)?, direct: false })
    }

    pub fn is_direct(&self) -> bool {
        self.direct
    }

    pub fn file(&self) -> &File {
        &self.file
    }
}

/// Positioned read of exactly `buf.len()` bytes.
pub(crate) fn read_exact_at(file: &File, buf: &mut [u8], offset: u64) -> io::Result<()> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::FileExt;
        file.read_exact_at(buf, offset)
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::FileExt;
        let mut done = 0;
        while done < buf.len() {
            let n = file.seek_read(&mut buf[done..], offset + done as u64)?;
            if n == 0 {
                return Err(io::ErrorKind::UnexpectedEof.into());
            }
            done += n;
        }
        Ok(())
    }
}

/// Drops `file`'s pages from the page cache (Linux; a no-op elsewhere).
/// Called after writing an index and after loading its RAM sections, so
/// that building or opening a large index does not evict other data.
pub(crate) fn drop_cache(file: &File) {
    #[cfg(target_os = "linux")]
    {
        use std::os::fd::AsRawFd;
        // SAFETY: a plain syscall on a descriptor that `file` keeps open; it
        // only gives the kernel advice and touches no memory of ours.
        unsafe { libc::posix_fadvise(file.as_raw_fd(), 0, 0, libc::POSIX_FADV_DONTNEED) };
    }
    #[cfg(not(target_os = "linux"))]
    let _ = file;
}

/// Bytes free for an unprivileged writer on the file system holding `dir`
/// (Linux; `None` elsewhere or on error).
pub(crate) fn free_space(dir: &std::path::Path) -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        use std::os::unix::ffi::OsStrExt;
        let path = std::ffi::CString::new(dir.as_os_str().as_bytes()).ok()?;
        let mut st = std::mem::MaybeUninit::<libc::statvfs>::uninit();
        // SAFETY: `path` is NUL-terminated and `st` is a valid out-pointer;
        // statvfs fills it completely when it returns 0.
        let st = unsafe {
            if libc::statvfs(path.as_ptr(), st.as_mut_ptr()) != 0 {
                return None;
            }
            st.assume_init()
        };
        Some((st.f_bavail as u64).saturating_mul(st.f_frsize as u64))
    }
    #[cfg(not(target_os = "linux"))]
    {
        let _ = dir;
        None
    }
}

/// Heap memory whose usable part starts on an `ALIGN` boundary. Never
/// reallocated after creation, so pointers into it stay valid.
struct AlignedBuf {
    raw: Vec<u8>,
    start: usize,
    len: usize,
}

impl AlignedBuf {
    fn new(len: usize) -> Self {
        let raw = vec![0u8; len + ALIGN];
        let start = raw.as_ptr().align_offset(ALIGN);
        assert!(start < ALIGN, "cannot align a byte buffer");
        Self { raw, start, len }
    }

    fn bytes(&self) -> &[u8] {
        &self.raw[self.start..self.start + self.len]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        &mut self.raw[self.start..self.start + self.len]
    }
}

enum Backend {
    Pread,
    #[cfg(target_os = "linux")]
    Uring(Box<io_uring::IoUring>),
}

/// Reads batches of blocks into its own buffers: request `i` of a batch
/// lands in slot `i`. One reader per thread; it is not shared.
pub(crate) struct Reader {
    // Field order matters: the ring is dropped before the buffers it may
    // write into.
    backend: Backend,
    buf: AlignedBuf,
    slot_bytes: usize,
    lens: [usize; DEPTH],
    /// Set if a batch failed while reads may still be in flight. The reader
    /// then refuses further work and leaks its buffers instead of freeing
    /// memory the kernel could still write to.
    poisoned: bool,
}

impl Reader {
    pub fn new(mode: IoMode) -> io::Result<Self> {
        let backend = match mode {
            IoMode::Pread => Backend::Pread,
            IoMode::Auto | IoMode::Uring => match uring() {
                Ok(b) => b,
                Err(_) if mode == IoMode::Auto => Backend::Pread,
                Err(e) => return Err(e),
            },
        };
        Ok(Self { backend, buf: AlignedBuf::new(0), slot_bytes: 0, lens: [0; DEPTH], poisoned: false })
    }

    /// "io_uring" or "pread".
    pub fn kind(&self) -> &'static str {
        match self.backend {
            Backend::Pread => "pread",
            #[cfg(target_os = "linux")]
            Backend::Uring(_) => "io_uring",
        }
    }

    /// Makes every slot at least `bytes` long (a multiple of `ALIGN`).
    pub fn reserve(&mut self, bytes: usize) {
        if bytes > self.slot_bytes && !self.poisoned {
            self.slot_bytes = bytes.div_ceil(ALIGN) * ALIGN;
            self.buf = AlignedBuf::new(self.slot_bytes * DEPTH);
        }
    }

    /// Bytes read into slot `i` by the last batch.
    pub fn slot(&self, i: usize) -> &[u8] {
        &self.buf.bytes()[i * self.slot_bytes..][..self.lens[i]]
    }

    /// Reads every `(offset, len)` in `reqs` into slots `0..reqs.len()` and
    /// returns when all are complete. Offsets and lengths must be multiples
    /// of `ALIGN`, lengths at most the reserved slot size, and
    /// `reqs.len() <= DEPTH`.
    pub fn read(&mut self, file: &DataFile, reqs: &[(u64, usize)]) -> io::Result<()> {
        if self.poisoned {
            return Err(io::Error::other("reader disabled after an io_uring failure"));
        }
        assert!(reqs.len() <= DEPTH, "too many reads in one batch");
        for (i, &(offset, len)) in reqs.iter().enumerate() {
            assert!(
                len <= self.slot_bytes && len % ALIGN == 0 && offset % ALIGN as u64 == 0,
                "unaligned or oversized read"
            );
            self.lens[i] = len;
        }
        match &mut self.backend {
            Backend::Pread => {
                for (i, &(offset, len)) in reqs.iter().enumerate() {
                    let slot = &mut self.buf.bytes_mut()[i * self.slot_bytes..][..len];
                    read_exact_at(file.file(), slot, offset)?;
                }
                Ok(())
            }
            #[cfg(target_os = "linux")]
            Backend::Uring(ring) => {
                match read_uring(ring, &mut self.buf, self.slot_bytes, file.file(), reqs) {
                    Ok(()) => Ok(()),
                    Err(UringError::Done(e)) => Err(e),
                    Err(UringError::InFlight(e)) => {
                        self.poisoned = true;
                        Err(e)
                    }
                }
            }
        }
    }
}

impl Drop for Reader {
    fn drop(&mut self) {
        if self.poisoned {
            // The kernel may still write into these buffers: never free them.
            std::mem::forget(std::mem::take(&mut self.buf.raw));
        }
    }
}

#[cfg(target_os = "linux")]
fn uring() -> io::Result<Backend> {
    Ok(Backend::Uring(Box::new(io_uring::IoUring::new(DEPTH as u32)?)))
}

#[cfg(not(target_os = "linux"))]
fn uring() -> io::Result<Backend> {
    Err(io::Error::new(io::ErrorKind::Unsupported, "io_uring needs Linux"))
}

#[cfg(target_os = "linux")]
enum UringError {
    /// Every read has finished; the batch failed.
    Done(io::Error),
    /// Reads may still be in flight.
    InFlight(io::Error),
}

#[cfg(target_os = "linux")]
fn read_uring(
    ring: &mut io_uring::IoUring,
    buf: &mut AlignedBuf,
    slot_bytes: usize,
    file: &File,
    reqs: &[(u64, usize)],
) -> Result<(), UringError> {
    use io_uring::{opcode, types};
    use std::os::fd::AsRawFd;

    let n = reqs.len();
    if n == 0 {
        return Ok(());
    }
    let fd = types::Fd(file.as_raw_fd());
    {
        let mut sq = ring.submission();
        if sq.capacity() - sq.len() < n {
            return Err(UringError::Done(io::Error::other("io_uring submission queue is full")));
        }
        let base = buf.bytes_mut().as_mut_ptr();
        for (i, &(offset, len)) in reqs.iter().enumerate() {
            // SAFETY: slot `i` is `slot_bytes >= len` bytes inside `buf`
            // (checked by the caller), so the pointer stays in bounds.
            let ptr = unsafe { base.add(i * slot_bytes) };
            let entry = opcode::Read::new(fd, ptr, len as u32).offset(offset).build().user_data(i as u64);
            // SAFETY: the kernel writes `len` bytes at `ptr`, which stays
            // allocated and unmoved: `buf` is never reallocated while reads
            // are in flight, this function does not return before every
            // completion is reaped (or the reader is poisoned and leaks its
            // buffers), and the descriptor outlives the call. Capacity was
            // checked above, so the push cannot fail halfway.
            unsafe { sq.push(&entry) }.map_err(|_| UringError::Done(io::Error::other("io_uring push failed")))?;
        }
    }

    let mut done = 0;
    let mut short = 0u64; // bit i: request i returned fewer bytes than asked
    let mut first_err: Option<io::Error> = None;
    let mut enter_failures = 0;
    while done < n {
        match ring.submit_and_wait(n - done) {
            Ok(_) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => {
                // EAGAIN / EBUSY are transient. Anything that persists leaves
                // the reads' state unknown, so the buffers must stay alive.
                enter_failures += 1;
                if enter_failures > 1000 {
                    return Err(UringError::InFlight(e));
                }
                std::thread::yield_now();
            }
        }
        for cqe in ring.completion() {
            done += 1;
            let i = cqe.user_data() as usize;
            let res = cqe.result();
            if i >= n {
                first_err.get_or_insert_with(|| io::Error::other("unexpected io_uring completion"));
            } else if res < 0 {
                first_err.get_or_insert_with(|| io::Error::from_raw_os_error(-res));
            } else if (res as usize) < reqs[i].1 {
                short |= 1 << i;
            }
        }
    }
    if let Some(e) = first_err {
        return Err(UringError::Done(e));
    }
    // A short read (rare; e.g. a signal) is redone in full synchronously.
    while short != 0 {
        let i = short.trailing_zeros() as usize;
        short &= short - 1;
        let (offset, len) = reqs[i];
        let slot = &mut buf.bytes_mut()[i * slot_bytes..][..len];
        read_exact_at(file, slot, offset).map_err(UringError::Done)?;
    }
    Ok(())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::io::Write;

    pub(crate) fn temp_path(name: &str) -> std::path::PathBuf {
        // Under target/ rather than /tmp, which is often tmpfs (no O_DIRECT).
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../target/test-tmp");
        std::fs::create_dir_all(&dir).unwrap();
        dir.join(format!("{name}-{}", std::process::id()))
    }

    #[test]
    fn every_backend_reads_the_same_bytes() {
        let path = temp_path("io-backends");
        let data: Vec<u8> = (0..20 * ALIGN).map(|i| (i * 31 % 251) as u8).collect();
        std::fs::File::create(&path).unwrap().write_all(&data).unwrap();
        let reqs: Vec<(u64, usize)> = vec![(0, ALIGN), (5 * ALIGN as u64, 2 * ALIGN), (19 * ALIGN as u64, ALIGN)];
        for direct in [false, true] {
            let file = DataFile::open(&path, direct).unwrap();
            for mode in [IoMode::Pread, IoMode::Auto] {
                let mut r = Reader::new(mode).unwrap();
                r.reserve(2 * ALIGN);
                r.read(&file, &reqs).unwrap();
                for (i, &(off, len)) in reqs.iter().enumerate() {
                    assert_eq!(r.slot(i), &data[off as usize..off as usize + len], "{} direct {direct}", r.kind());
                }
                // Reading past the end is an error, not a short slot.
                assert!(r.read(&file, &[(20 * ALIGN as u64, ALIGN)]).is_err(), "{}", r.kind());
                // The reader still works after a failed batch.
                r.read(&file, &reqs[..1]).unwrap();
                assert_eq!(r.slot(0), &data[..ALIGN]);
            }
        }
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn buffers_are_aligned() {
        for len in [0usize, 1, 4096, 10_000] {
            let b = AlignedBuf::new(len);
            assert_eq!(b.bytes().as_ptr() as usize % ALIGN, 0);
            assert_eq!(b.bytes().len(), len);
        }
    }
}
