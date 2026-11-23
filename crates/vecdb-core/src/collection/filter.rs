//! Attribute filters for search.
//!
//! A filter is a tree over the collection's declared fields. A row with
//! several values for a field matches a condition on it when any of its
//! values does; a row with none matches no condition on it (and so matches
//! its `Not`).
//!
//! - `Eq` / `In`: a value equal to the one given / to any of the set (an
//!   empty set matches nothing). Integers and keywords.
//! - `Range`: an integer value in `min..=max` (empty when `min > max`).
//! - `And` / `Or`: all / any of the children; an empty `And` matches every
//!   row, an empty `Or` none.
//! - `Not`: the rows the child does not match.
//!
//! An unknown field, or a value of the wrong type for its field, is an
//! error, not an empty result.

use std::ops::{Bound, RangeBounds};

use super::bitmap::Bitmap;
use super::meta::{Attrs, Field, FieldType, Value};
use crate::error::{Error, Result};

/// Deepest filter tree accepted.
pub const MAX_FILTER_DEPTH: usize = 32;
/// Most nodes and values in one filter.
pub const MAX_FILTER_TERMS: usize = 1 << 16;

#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    Eq(String, Value),
    In(String, Vec<Value>),
    /// Integers in `min..=max`.
    Range { field: String, min: i64, max: i64 },
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

impl Filter {
    pub fn int_eq(field: &str, v: i64) -> Self {
        Filter::Eq(field.into(), Value::Int(v))
    }

    pub fn keyword_eq(field: &str, v: &str) -> Self {
        Filter::Eq(field.into(), Value::Keyword(v.into()))
    }

    pub fn keyword_in<S: AsRef<str>>(field: &str, vs: &[S]) -> Self {
        Filter::In(field.into(), vs.iter().map(|s| Value::Keyword(s.as_ref().into())).collect())
    }

    pub fn int_in(field: &str, vs: &[i64]) -> Self {
        Filter::In(field.into(), vs.iter().map(|&v| Value::Int(v)).collect())
    }

    /// Integers in `range` (any Rust range: `a..b`, `a..=b`, `a..`, `..b`).
    pub fn range(field: &str, range: impl RangeBounds<i64>) -> Self {
        let (mut min, mut max) = (i64::MIN, i64::MAX);
        let mut empty = false;
        match range.start_bound() {
            Bound::Included(&a) => min = a,
            Bound::Excluded(&a) => match a.checked_add(1) {
                Some(a) => min = a,
                None => empty = true,
            },
            Bound::Unbounded => {}
        }
        match range.end_bound() {
            Bound::Included(&b) => max = b,
            Bound::Excluded(&b) => match b.checked_sub(1) {
                Some(b) => max = b,
                None => empty = true,
            },
            Bound::Unbounded => {}
        }
        if empty {
            (min, max) = (1, 0);
        }
        Filter::Range { field: field.into(), min, max }
    }

    pub fn and(fs: impl IntoIterator<Item = Filter>) -> Self {
        Filter::And(fs.into_iter().collect())
    }

    pub fn or(fs: impl IntoIterator<Item = Filter>) -> Self {
        Filter::Or(fs.into_iter().collect())
    }

    #[allow(clippy::should_implement_trait)]
    pub fn not(f: Filter) -> Self {
        Filter::Not(Box::new(f))
    }

    /// Whether a row with `attrs` matches, evaluated directly (the
    /// reference for the indexed evaluation).
    pub fn matches(&self, attrs: &Attrs) -> bool {
        fn values<'a>(attrs: &'a Attrs, field: &'a str) -> impl Iterator<Item = &'a Value> + 'a {
            attrs.entries().iter().filter(move |(n, _)| n == field).map(|(_, v)| v)
        }
        match self {
            Filter::Eq(f, v) => values(attrs, f).any(|x| x == v),
            Filter::In(f, vs) => values(attrs, f).any(|x| vs.contains(x)),
            Filter::Range { field, min, max } => {
                values(attrs, field).any(|x| matches!(x, Value::Int(i) if (min..=max).contains(&i)))
            }
            Filter::And(fs) => fs.iter().all(|f| f.matches(attrs)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(attrs)),
            Filter::Not(f) => !f.matches(attrs),
        }
    }

    /// Checks the filter against `fields` and resolves field names.
    pub(crate) fn compile(&self, fields: &[Field]) -> Result<Compiled> {
        let mut terms = 0;
        let c = self.compile_at(fields, 1, &mut terms)?;
        Ok(c.simplify())
    }

    fn compile_at(&self, fields: &[Field], depth: usize, terms: &mut usize) -> Result<Compiled> {
        if depth > MAX_FILTER_DEPTH {
            return Err(Error::InvalidArgument(format!("filter deeper than {MAX_FILTER_DEPTH}")));
        }
        *terms += 1;
        if *terms > MAX_FILTER_TERMS {
            return Err(Error::InvalidArgument(format!("filter with more than {MAX_FILTER_TERMS} terms")));
        }
        let field = |name: &str| -> Result<(usize, FieldType)> {
            fields
                .iter()
                .position(|f| f.name == name)
                .map(|i| (i, fields[i].ty))
                .ok_or_else(|| Error::InvalidArgument(format!("filter on unknown field {name:?}")))
        };
        let leaf = |name: &str, vs: &[Value], terms: &mut usize| -> Result<Compiled> {
            let (i, ty) = field(name)?;
            *terms += vs.len();
            if *terms > MAX_FILTER_TERMS {
                return Err(Error::InvalidArgument(format!("filter with more than {MAX_FILTER_TERMS} terms")));
            }
            if let Some(v) = vs.iter().find(|v| v.ty() != ty) {
                return Err(Error::InvalidArgument(format!("field {name:?} holds {ty:?} values, not {v:?}")));
            }
            Ok(match ty {
                FieldType::Int => Compiled::Ints {
                    field: i,
                    ranges: vs
                        .iter()
                        .map(|v| match v {
                            Value::Int(x) => (*x, *x),
                            Value::Keyword(_) => unreachable!("checked"),
                        })
                        .collect(),
                },
                FieldType::Keyword => Compiled::Keywords {
                    field: i,
                    keys: vs
                        .iter()
                        .map(|v| match v {
                            Value::Keyword(s) => s.as_bytes().to_vec(),
                            Value::Int(_) => unreachable!("checked"),
                        })
                        .collect(),
                },
            })
        };
        Ok(match self {
            Filter::Eq(f, v) => leaf(f, std::slice::from_ref(v), terms)?,
            Filter::In(f, vs) => leaf(f, vs, terms)?,
            Filter::Range { field: f, min, max } => {
                let (i, ty) = field(f)?;
                if ty != FieldType::Int {
                    return Err(Error::InvalidArgument(format!("range on keyword field {f:?}")));
                }
                Compiled::Ints { field: i, ranges: if min <= max { vec![(*min, *max)] } else { Vec::new() } }
            }
            Filter::And(fs) => {
                Compiled::And(fs.iter().map(|f| f.compile_at(fields, depth + 1, terms)).collect::<Result<_>>()?)
            }
            Filter::Or(fs) => {
                Compiled::Or(fs.iter().map(|f| f.compile_at(fields, depth + 1, terms)).collect::<Result<_>>()?)
            }
            Filter::Not(f) => Compiled::Not(Box::new(f.compile_at(fields, depth + 1, terms)?)),
        })
    }
}

/// A filter with field numbers in place of names.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Compiled {
    /// Every row / no row.
    All,
    Nothing,
    /// A value in any of the inclusive ranges.
    Ints { field: usize, ranges: Vec<(i64, i64)> },
    /// A value equal to any of the keys.
    Keywords { field: usize, keys: Vec<Vec<u8>> },
    And(Vec<Compiled>),
    Or(Vec<Compiled>),
    Not(Box<Compiled>),
}

impl Compiled {
    /// Folds constant parts: empty leaves become `Nothing`, and `All` /
    /// `Nothing` propagate through `And`, `Or` and `Not`.
    fn simplify(self) -> Compiled {
        match self {
            Compiled::Ints { ranges, .. } if ranges.is_empty() => Compiled::Nothing,
            Compiled::Keywords { keys, .. } if keys.is_empty() => Compiled::Nothing,
            Compiled::And(cs) => {
                let mut out = Vec::new();
                for c in cs.into_iter().map(Compiled::simplify) {
                    match c {
                        Compiled::All => {}
                        Compiled::Nothing => return Compiled::Nothing,
                        c => out.push(c),
                    }
                }
                match out.len() {
                    0 => Compiled::All,
                    1 => out.pop().unwrap(),
                    _ => Compiled::And(out),
                }
            }
            Compiled::Or(cs) => {
                let mut out = Vec::new();
                for c in cs.into_iter().map(Compiled::simplify) {
                    match c {
                        Compiled::Nothing => {}
                        Compiled::All => return Compiled::All,
                        c => out.push(c),
                    }
                }
                match out.len() {
                    0 => Compiled::Nothing,
                    1 => out.pop().unwrap(),
                    _ => Compiled::Or(out),
                }
            }
            Compiled::Not(c) => match c.simplify() {
                Compiled::All => Compiled::Nothing,
                Compiled::Nothing => Compiled::All,
                c => Compiled::Not(Box::new(c)),
            },
            c => c,
        }
    }

    /// The positions of `src` that match.
    pub fn eval(&self, src: &dyn ColumnSource) -> Bitmap {
        let n = src.positions();
        match self {
            Compiled::All => Bitmap::full(n),
            Compiled::Nothing => Bitmap::empty(n),
            Compiled::Ints { field, ranges } => src.ints(*field, ranges),
            Compiled::Keywords { field, keys } => src.keywords(*field, keys),
            Compiled::And(cs) => {
                let mut b = cs[0].eval(src);
                for c in &cs[1..] {
                    b.and(&c.eval(src));
                }
                b
            }
            Compiled::Or(cs) => {
                let mut b = cs[0].eval(src);
                for c in &cs[1..] {
                    b.or(&c.eval(src));
                }
                b
            }
            Compiled::Not(c) => {
                let mut b = c.eval(src);
                b.not();
                b
            }
        }
    }
}

/// Attribute columns of one segment, as a filter reads them.
pub(crate) trait ColumnSource {
    fn positions(&self) -> usize;
    /// The positions with an integer of `field` in any of the inclusive
    /// `ranges`.
    fn ints(&self, field: usize, ranges: &[(i64, i64)]) -> Bitmap;
    /// The positions with any of `keys` in `field`.
    fn keywords(&self, field: usize, keys: &[Vec<u8>]) -> Bitmap;
}

/// `ranges` sorted and merged: disjoint, ascending, not adjacent.
pub(crate) fn merge_ranges(ranges: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut r: Vec<(i64, i64)> = ranges.iter().copied().filter(|(lo, hi)| lo <= hi).collect();
    r.sort_unstable();
    let mut out: Vec<(i64, i64)> = Vec::with_capacity(r.len());
    for (lo, hi) in r {
        match out.last_mut() {
            Some(last) if last.1 == i64::MAX || lo <= last.1 + 1 => last.1 = last.1.max(hi),
            _ => out.push((lo, hi)),
        }
    }
    out
}
