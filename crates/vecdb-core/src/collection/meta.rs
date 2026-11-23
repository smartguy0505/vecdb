//! Row attributes (metadata). A collection declares its fields up front,
//! each with a type; a row carries any number of values for any of them
//! (several keywords of one field act as a set of tags).

use super::bytes::{Dec, Enc};
use crate::error::{Error, Result};

/// Longest field name or keyword, in bytes.
pub const MAX_STR: usize = 1024;
/// Most fields per collection.
pub const MAX_FIELDS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FieldType {
    /// Signed 64-bit integers: equality, set membership and ranges.
    Int,
    /// Strings matched exactly: equality and set membership.
    Keyword,
}

impl FieldType {
    pub(crate) fn code(self) -> u8 {
        match self {
            FieldType::Int => 1,
            FieldType::Keyword => 2,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            1 => Some(FieldType::Int),
            2 => Some(FieldType::Keyword),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Field {
    pub name: String,
    pub ty: FieldType,
}

impl Field {
    pub fn int(name: &str) -> Self {
        Self { name: name.into(), ty: FieldType::Int }
    }

    pub fn keyword(name: &str) -> Self {
        Self { name: name.into(), ty: FieldType::Keyword }
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Value {
    Int(i64),
    Keyword(String),
}

impl Value {
    pub fn ty(&self) -> FieldType {
        match self {
            Value::Int(_) => FieldType::Int,
            Value::Keyword(_) => FieldType::Keyword,
        }
    }
}

/// The attributes of one row: (field name, value) pairs. A field may appear
/// several times (a set of tags); order does not matter.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Attrs {
    entries: Vec<(String, Value)>,
}

/// A row without attributes.
pub static NO_ATTRS: Attrs = Attrs { entries: Vec::new() };

impl Attrs {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn int(mut self, field: &str, v: i64) -> Self {
        self.entries.push((field.into(), Value::Int(v)));
        self
    }

    pub fn keyword(mut self, field: &str, v: &str) -> Self {
        self.entries.push((field.into(), Value::Keyword(v.into())));
        self
    }

    pub fn entries(&self) -> &[(String, Value)] {
        &self.entries
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Checks every entry against the collection's fields.
    pub(crate) fn validate(&self, fields: &[Field]) -> Result<()> {
        if self.entries.len() > u16::MAX as usize {
            return Err(Error::InvalidArgument("too many attribute values for one row".into()));
        }
        for (name, v) in &self.entries {
            let f = fields
                .iter()
                .find(|f| f.name == *name)
                .ok_or_else(|| Error::InvalidArgument(format!("unknown field {name:?}")))?;
            if f.ty != v.ty() {
                return Err(Error::InvalidArgument(format!("field {name:?} holds {:?} values", f.ty)));
            }
            if let Value::Keyword(s) = v
                && s.len() > MAX_STR
            {
                return Err(Error::InvalidArgument(format!("keyword longer than {MAX_STR} bytes")));
            }
        }
        Ok(())
    }

    pub(crate) fn encoded_len(&self) -> usize {
        2 + self
            .entries
            .iter()
            .map(|(n, v)| {
                2 + n.len()
                    + 1
                    + match v {
                        Value::Int(_) => 8,
                        Value::Keyword(s) => 2 + s.len(),
                    }
            })
            .sum::<usize>()
    }

    pub(crate) fn encode(&self, e: &mut Enc) {
        e.u16(self.entries.len() as u16);
        for (name, v) in &self.entries {
            e.str(name);
            match v {
                Value::Int(i) => {
                    e.u8(FieldType::Int.code());
                    e.i64(*i);
                }
                Value::Keyword(s) => {
                    e.u8(FieldType::Keyword.code());
                    e.str(s);
                }
            }
        }
    }

    pub(crate) fn decode(bytes: &[u8]) -> Result<Self> {
        let mut d = Dec::new(bytes, "attributes");
        let mut entries = Vec::new();
        for _ in 0..d.u16()? {
            let name = d.str()?.to_string();
            let v = match FieldType::from_code(d.u8()?) {
                Some(FieldType::Int) => Value::Int(d.i64()?),
                Some(FieldType::Keyword) => Value::Keyword(d.str()?.to_string()),
                None => return Err(d.corrupt("unknown value type")),
            };
            entries.push((name, v));
        }
        d.done()?;
        Ok(Self { entries })
    }
}

pub(crate) fn validate_fields(fields: &[Field]) -> Result<()> {
    if fields.len() > MAX_FIELDS {
        return Err(Error::InvalidArgument(format!("at most {MAX_FIELDS} fields")));
    }
    for (i, f) in fields.iter().enumerate() {
        if f.name.is_empty() || f.name.len() > MAX_STR {
            return Err(Error::InvalidArgument("field names must be 1 to 1024 bytes".into()));
        }
        if fields[..i].iter().any(|g| g.name == f.name) {
            return Err(Error::InvalidArgument(format!("field {:?} declared twice", f.name)));
        }
    }
    Ok(())
}
