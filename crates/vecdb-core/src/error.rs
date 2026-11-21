use std::fmt;

#[derive(Debug)]
pub enum Error {
    DimensionMismatch { expected: usize, got: usize },
    /// A vector or query contained NaN or +/-infinity.
    NonFiniteValue,
    DuplicateId(u64),
    InvalidArgument(String),
    /// On-disk data failed a structural or checksum check.
    Corrupt(String),
    /// On-disk data is valid but written by a format version we cannot read.
    Unsupported(String),
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Error::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Error::NonFiniteValue => write!(f, "vector contains NaN or infinity"),
            Error::DuplicateId(id) => write!(f, "duplicate id {id}"),
            Error::InvalidArgument(m) => write!(f, "invalid argument: {m}"),
            Error::Corrupt(m) => write!(f, "corrupt data: {m}"),
            Error::Unsupported(m) => write!(f, "unsupported: {m}"),
            Error::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Error::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for Error {
    fn from(e: std::io::Error) -> Self {
        Error::Io(e)
    }
}
