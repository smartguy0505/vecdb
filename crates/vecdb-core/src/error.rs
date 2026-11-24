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
    /// No collection of that name.
    NotFound(String),
    /// The collection is there already.
    AlreadyExists(String),
    /// Not possible now; may be later.
    Unavailable(String),
    /// Stopped through a [`Cancel`](crate::Cancel).
    Cancelled,
    Io(std::io::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

impl Error {
    /// A copy, for reporting one failure to several callers (an I/O error
    /// keeps its kind and message).
    pub fn duplicate(&self) -> Error {
        match self {
            Error::DimensionMismatch { expected, got } => Error::DimensionMismatch { expected: *expected, got: *got },
            Error::NonFiniteValue => Error::NonFiniteValue,
            Error::DuplicateId(id) => Error::DuplicateId(*id),
            Error::InvalidArgument(m) => Error::InvalidArgument(m.clone()),
            Error::Corrupt(m) => Error::Corrupt(m.clone()),
            Error::Unsupported(m) => Error::Unsupported(m.clone()),
            Error::NotFound(m) => Error::NotFound(m.clone()),
            Error::AlreadyExists(m) => Error::AlreadyExists(m.clone()),
            Error::Unavailable(m) => Error::Unavailable(m.clone()),
            Error::Cancelled => Error::Cancelled,
            Error::Io(e) => Error::Io(std::io::Error::new(e.kind(), e.to_string())),
        }
    }
}

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
            Error::NotFound(m) => write!(f, "not found: {m}"),
            Error::AlreadyExists(m) => write!(f, "already exists: {m}"),
            Error::Unavailable(m) => write!(f, "unavailable: {m}"),
            Error::Cancelled => write!(f, "cancelled"),
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
