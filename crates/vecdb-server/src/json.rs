//! The JSON of requests and replies, and what it means to the engine.
//!
//! Unknown keys, keys given twice and values of the wrong type are errors,
//! so that a typo does not quietly change what a request does.

use std::collections::HashSet;
use std::fmt;
use std::marker::PhantomData;
use std::ops::Bound;

use serde::de::{self, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::ser::{SerializeMap, Serializer};
use serde::{Deserialize, Serialize};
use vecdb_core::collection::SearchStats;
use vecdb_core::quant::Codec;
use vecdb_core::{
    Attrs, CollectionConfig, Error, Fetch, Field, FieldType, Filter, FilterMode, Layout, Metric, Op, Row,
    SearchOptions, SearchRequest, Value,
};

use crate::base64;

type Result<T> = std::result::Result<T, Error>;

fn invalid<T>(m: impl Into<String>) -> Result<T> {
    Err(Error::InvalidArgument(m.into()))
}

/// A vector: an array of numbers, or a base64 string of little-endian `f32`s.
#[derive(Debug, PartialEq)]
pub struct Vector(pub Vec<f32>);

impl<'de> Deserialize<'de> for Vector {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Vector;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an array of numbers, or a base64 string of little-endian f32 values")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Vector, A::Error> {
                let mut v = Vec::new();
                while let Some(x) = seq.next_element::<f32>()? {
                    v.push(x);
                }
                Ok(Vector(v))
            }

            fn visit_str<E: de::Error>(self, s: &str) -> std::result::Result<Vector, E> {
                let bytes = base64::decode(s).ok_or_else(|| E::custom("vector: not valid base64"))?;
                let (floats, rest) = bytes.as_chunks::<4>();
                if !rest.is_empty() {
                    return Err(E::custom("vector: base64 of a length that is not a multiple of 4 bytes"));
                }
                Ok(Vector(floats.iter().map(|b| f32::from_le_bytes(*b)).collect()))
            }
        }
        d.deserialize_any(V)
    }
}

/// An attribute value: an integer or a string.
#[derive(Debug)]
pub struct Scalar(pub Value);

const SCALAR: &str = "an integer (64-bit, signed) or a string";

struct ScalarVisitor;

impl Visitor<'_> for ScalarVisitor {
    type Value = Scalar;

    fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(SCALAR)
    }

    fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Scalar, E> {
        Ok(Scalar(Value::Int(v)))
    }

    fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Scalar, E> {
        match i64::try_from(v) {
            Ok(v) => self.visit_i64(v),
            Err(_) => Err(E::invalid_value(de::Unexpected::Unsigned(v), &self)),
        }
    }

    fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Scalar, E> {
        Ok(Scalar(Value::Keyword(v.into())))
    }
}

impl<'de> Deserialize<'de> for Scalar {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        d.deserialize_any(ScalarVisitor)
    }
}

/// One attribute value, or an array of them.
#[derive(Debug)]
pub struct Scalars(Vec<Value>);

/// Most values of one field of a row: they take some 40 times the memory of
/// their JSON.
const MAX_VALUES: usize = 1024;

impl<'de> Deserialize<'de> for Scalars {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V;
        impl<'de> Visitor<'de> for V {
            type Value = Scalars;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                write!(f, "{SCALAR}, or an array of them")
            }

            fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> std::result::Result<Scalars, A::Error> {
                let mut v = Vec::new();
                while let Some(Scalar(x)) = seq.next_element()? {
                    if v.len() == MAX_VALUES {
                        return Err(de::Error::custom(format!("at most {MAX_VALUES} values of a field")));
                    }
                    v.push(x);
                }
                Ok(Scalars(v))
            }

            fn visit_i64<E: de::Error>(self, v: i64) -> std::result::Result<Scalars, E> {
                ScalarVisitor.visit_i64(v).map(|s| Scalars(vec![s.0]))
            }

            fn visit_u64<E: de::Error>(self, v: u64) -> std::result::Result<Scalars, E> {
                ScalarVisitor.visit_u64(v).map(|s| Scalars(vec![s.0]))
            }

            fn visit_str<E: de::Error>(self, v: &str) -> std::result::Result<Scalars, E> {
                ScalarVisitor.visit_str(v).map(|s| Scalars(vec![s.0]))
            }
        }
        d.deserialize_any(V)
    }
}

/// An object, as its keys and values in order.
#[derive(Debug)]
pub struct Pairs<T>(pub Vec<(String, T)>);

impl<T> Default for Pairs<T> {
    fn default() -> Self {
        Self(Vec::new())
    }
}

impl<'de, T: Deserialize<'de>> Deserialize<'de> for Pairs<T> {
    fn deserialize<D: Deserializer<'de>>(d: D) -> std::result::Result<Self, D::Error> {
        struct V<T>(PhantomData<T>);
        impl<'de, T: Deserialize<'de>> Visitor<'de> for V<T> {
            type Value = Pairs<T>;

            fn expecting(&self, f: &mut fmt::Formatter) -> fmt::Result {
                f.write_str("an object")
            }

            fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> std::result::Result<Pairs<T>, A::Error> {
                let (mut v, mut seen) = (Vec::new(), HashSet::new());
                while let Some(key) = map.next_key::<String>()? {
                    if !seen.insert(key.clone()) {
                        return Err(de::Error::custom(format!("{key:?} given twice")));
                    }
                    v.push((key, map.next_value()?));
                }
                Ok(Pairs(v))
            }
        }
        d.deserialize_map(V(PhantomData))
    }
}

impl Pairs<Scalars> {
    fn attrs(self) -> Attrs {
        let mut a = Attrs::new();
        for (name, values) in self.0 {
            for v in values.0 {
                a = match v {
                    Value::Int(i) => a.int(&name, i),
                    Value::Keyword(s) => a.keyword(&name, &s),
                };
            }
        }
        a
    }
}

/// Settings of a new collection; what is left out has its default.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Create {
    pub dim: usize,
    pub metric: String,
    pub shards: Option<usize>,
    /// Name to "int" or "keyword".
    #[serde(default)]
    pub fields: Pairs<String>,
    pub seal_bytes: Option<u64>,
    pub index_bytes: Option<u64>,
    pub compact_deleted: Option<f32>,
    pub max_indexed_segments: Option<usize>,
    #[serde(default)]
    pub index: Index,
}

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Index {
    pub max_degree: Option<usize>,
    pub build_list: Option<usize>,
    pub alpha: Option<f32>,
    pub passes: Option<usize>,
    pub pq_m: Option<usize>,
    pub codec: Option<String>,
    pub layout: Option<String>,
}

/// Largest settings a client may ask for, beyond what the engine checks: ones
/// that would let a collection take the machine's memory or CPUs.
const MAX_SEAL_BYTES: u64 = 4 << 30;
const MAX_INDEX_BYTES: u64 = 64 << 30;
const MAX_BUILD_LIST: usize = 4096;
const MAX_PASSES: usize = 8;

impl Create {
    pub fn config(self) -> Result<CollectionConfig> {
        let Some(metric) = Metric::parse(&self.metric) else {
            return invalid(format!("metric {:?}: use \"l2\", \"ip\" or \"cosine\"", self.metric));
        };
        let mut c = CollectionConfig::new(self.dim, metric);
        let i = self.index;
        c.seal_bytes = self.seal_bytes.unwrap_or(c.seal_bytes);
        c.index_bytes = self.index_bytes.unwrap_or(c.index_bytes);
        c.compact_deleted = self.compact_deleted.unwrap_or(c.compact_deleted);
        c.max_indexed_segments = self.max_indexed_segments.unwrap_or(c.max_indexed_segments);
        c.index.max_degree = i.max_degree.unwrap_or(c.index.max_degree);
        c.index.build_list = i.build_list.unwrap_or(c.index.build_list);
        c.index.alpha = i.alpha.unwrap_or(c.index.alpha);
        c.index.passes = i.passes.unwrap_or(c.index.passes);
        c.index.pq_m = i.pq_m.unwrap_or(c.index.pq_m);
        if let Some(name) = i.codec {
            c.index.codec = match Codec::parse(&name) {
                Some(codec) => codec,
                None => return invalid(format!("codec {name:?}: use \"f32\", \"f16\" or \"bf16\"")),
            };
        }
        c.index.layout = match i.layout.as_deref() {
            None | Some("auto") => None,
            Some("inline") => Some(Layout::Inline),
            Some("separate") => Some(Layout::Separate),
            Some(other) => return invalid(format!("layout {other:?}: use \"auto\", \"inline\" or \"separate\"")),
        };
        for (name, ty) in self.fields.0 {
            c.fields.push(match ty.as_str() {
                "int" => Field::int(&name),
                "keyword" => Field::keyword(&name),
                _ => return invalid(format!("field {name:?}: type {ty:?}: use \"int\" or \"keyword\"")),
            });
        }
        // One vector may be larger than any of these.
        if c.seal_bytes > MAX_SEAL_BYTES || c.index_bytes > MAX_INDEX_BYTES {
            return invalid(format!("at most {MAX_SEAL_BYTES} seal_bytes and {MAX_INDEX_BYTES} index_bytes"));
        }
        if c.index.build_list > MAX_BUILD_LIST || c.index.passes > MAX_PASSES {
            return invalid(format!("at most a build_list of {MAX_BUILD_LIST} and {MAX_PASSES} passes"));
        }
        Ok(c)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowIn {
    pub id: u64,
    pub vector: Vector,
    #[serde(default)]
    pub attrs: Pairs<Scalars>,
}

impl RowIn {
    fn row(self) -> Row {
        Row { id: self.id, vector: self.vector.0, attrs: self.attrs.attrs() }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Upsert {
    pub rows: Vec<RowIn>,
}

impl Upsert {
    pub fn ops(self) -> Vec<Op> {
        self.rows.into_iter().map(|r| Op::Upsert(r.row())).collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Ids {
    pub ids: Vec<u64>,
}

impl Ids {
    pub fn deletes(self) -> Vec<Op> {
        self.ids.into_iter().map(|id| Op::Delete { id }).collect()
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "lowercase")]
pub enum OpIn {
    Upsert(RowIn),
    Delete(DeleteIn),
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DeleteIn {
    pub id: u64,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Write {
    pub ops: Vec<OpIn>,
}

impl Write {
    pub fn ops(self) -> Vec<Op> {
        let op = |op| match op {
            OpIn::Upsert(r) => Op::Upsert(r.row()),
            OpIn::Delete(d) => Op::Delete { id: d.id },
        };
        self.ops.into_iter().map(op).collect()
    }
}

/// A filter: `{"field": F, "eq": V}`, `{"field": F, "in": [V, ...]}`,
/// `{"field": F, "gte": N, "lt": N}` (any of gt, gte, lt, lte),
/// `{"and": [...]}`, `{"or": [...]}` or `{"not": {...}}`.
#[derive(Deserialize)]
#[serde(try_from = "Parts")]
pub struct FilterIn(pub Filter);

#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields, default)]
struct Parts {
    field: Option<String>,
    eq: Option<Scalar>,
    r#in: Option<Vec<Scalar>>,
    gt: Option<i64>,
    gte: Option<i64>,
    lt: Option<i64>,
    lte: Option<i64>,
    and: Option<Vec<FilterIn>>,
    or: Option<Vec<FilterIn>>,
    not: Option<Box<FilterIn>>,
}

impl TryFrom<Parts> for FilterIn {
    type Error = String;

    fn try_from(p: Parts) -> std::result::Result<Self, String> {
        let range = p.gt.is_some() || p.gte.is_some() || p.lt.is_some() || p.lte.is_some();
        let tests = [p.eq.is_some(), p.r#in.is_some(), range];
        let all = [p.and.is_some(), p.or.is_some(), p.not.is_some(), p.field.is_some()];
        let some = |v: &[bool]| v.iter().filter(|&&b| b).count();
        if some(&all) != 1 || some(&tests) != p.field.is_some() as usize {
            return Err("a filter is one of \"and\", \"or\", \"not\", or a \"field\" with one of \"eq\", \"in\" \
                        or a range"
                .into());
        }
        let filters = |fs: Vec<FilterIn>| fs.into_iter().map(|f| f.0).collect();
        if let Some(fs) = p.and {
            return Ok(Self(Filter::And(filters(fs))));
        }
        if let Some(fs) = p.or {
            return Ok(Self(Filter::Or(filters(fs))));
        }
        if let Some(f) = p.not {
            return Ok(Self(Filter::not(f.0)));
        }
        let field = p.field.expect("checked");
        if let Some(v) = p.eq {
            return Ok(Self(Filter::Eq(field, v.0)));
        }
        if let Some(vs) = p.r#in {
            return Ok(Self(Filter::In(field, vs.into_iter().map(|v| v.0).collect())));
        }
        let bound = |open: Option<i64>, closed: Option<i64>, names: &str| match (open, closed) {
            (None, None) => Ok(Bound::Unbounded),
            (Some(v), None) => Ok(Bound::Excluded(v)),
            (None, Some(v)) => Ok(Bound::Included(v)),
            (Some(_), Some(_)) => Err(format!("a filter may have one of {names}")),
        };
        let from = bound(p.gt, p.gte, "\"gt\" and \"gte\"")?;
        let to = bound(p.lt, p.lte, "\"lt\" and \"lte\"")?;
        Ok(Self(Filter::range(&field, (from, to))))
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub vector: Vector,
    pub k: Option<usize>,
    pub l: Option<usize>,
    pub beam_width: Option<usize>,
    pub rerank: Option<usize>,
    pub filter: Option<FilterIn>,
    pub filter_mode: Option<String>,
    #[serde(default)]
    pub with_attrs: bool,
    #[serde(default)]
    pub with_vectors: bool,
    #[serde(default)]
    pub with_stats: bool,
}

impl Search {
    /// The search, and whether its counters are wanted. `k`, `l` and the
    /// rows' bytes may be as large as these.
    pub fn request(self, max_k: usize, max_l: usize, max_row_bytes: usize) -> Result<(SearchRequest, bool)> {
        let d = SearchOptions::default();
        let k = self.k.unwrap_or(10);
        let l = self.l.unwrap_or(d.l);
        let rerank = self.rerank.unwrap_or(d.rerank);
        if !(1..=max_k).contains(&k) {
            return invalid(format!("k must be in 1..={max_k}"));
        }
        if !(1..=max_l).contains(&l) || rerank > max_l {
            return invalid(format!("l must be in 1..={max_l}, and rerank at most that"));
        }
        let beam_width = self.beam_width.unwrap_or(d.beam_width);
        if !(1..=vecdb_core::disk::MAX_BEAM_WIDTH).contains(&beam_width) {
            return invalid(format!("beam_width must be in 1..={}", vecdb_core::disk::MAX_BEAM_WIDTH));
        }
        let filter_mode = match self.filter_mode.as_deref() {
            None | Some("auto") => FilterMode::Auto,
            Some("graph") => FilterMode::Graph,
            Some("scan") => FilterMode::Scan,
            Some(m) => return invalid(format!("filter_mode {m:?}: use \"auto\", \"graph\" or \"scan\"")),
        };
        let filter = self.filter.map(|f| f.0);
        let options = SearchOptions { l, beam_width, rerank, filter, filter_mode, ..d };
        let fetch = Fetch { vector: self.with_vectors, attrs: self.with_attrs };
        Ok((SearchRequest { vector: self.vector.0, k, options, fetch, max_row_bytes }, self.with_stats))
    }
}

/// Attributes as an object: a field's value, or an array of them if there
/// are several.
struct AttrsOut<'a>(&'a Attrs);

impl Serialize for AttrsOut<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        struct One<'a>(&'a Value);
        impl Serialize for One<'_> {
            fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
                match self.0 {
                    Value::Int(i) => s.serialize_i64(*i),
                    Value::Keyword(k) => s.serialize_str(k),
                }
            }
        }
        let mut names: Vec<&str> = Vec::new();
        for (n, _) in self.0.entries() {
            if !names.contains(&n.as_str()) {
                names.push(n);
            }
        }
        let mut m = s.serialize_map(Some(names.len()))?;
        for name in names {
            let values: Vec<One> = self.0.entries().iter().filter(|(n, _)| n == name).map(|(_, v)| One(v)).collect();
            match &values[..] {
                [one] => m.serialize_entry(name, one)?,
                several => m.serialize_entry(name, several)?,
            }
        }
        m.end()
    }
}

/// A row in a reply.
#[derive(Serialize)]
pub struct RowOut<'a> {
    id: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    distance: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    vector: Option<&'a [f32]>,
    #[serde(skip_serializing_if = "Option::is_none")]
    attrs: Option<AttrsOut<'a>>,
}

impl<'a> RowOut<'a> {
    /// With what `fetch` asked for of `row`.
    pub fn new(id: u64, distance: Option<f32>, row: Option<&'a Row>, fetch: Fetch) -> Self {
        Self {
            id,
            distance,
            vector: row.filter(|_| fetch.vector).map(|r| &r.vector[..]),
            attrs: row.filter(|_| fetch.attrs).map(|r| AttrsOut(&r.attrs)),
        }
    }
}

#[derive(Serialize)]
pub struct StatsOut {
    segments: usize,
    scanned: usize,
    reads: usize,
    rounds: usize,
    expanded: usize,
    matched: usize,
    graph_segments: usize,
    scan_segments: usize,
    fallbacks: usize,
    pq_scanned: usize,
}

impl From<&SearchStats> for StatsOut {
    fn from(s: &SearchStats) -> Self {
        Self {
            segments: s.segments,
            scanned: s.scanned,
            reads: s.reads,
            rounds: s.rounds,
            expanded: s.expanded,
            matched: s.matched,
            graph_segments: s.graph_segments,
            scan_segments: s.scan_segments,
            fallbacks: s.fallbacks,
            pq_scanned: s.pq_scanned,
        }
    }
}

#[derive(Serialize)]
pub struct RowsOut<'a> {
    pub rows: Vec<RowOut<'a>>,
}

#[derive(Serialize)]
pub struct FoundOut<'a> {
    pub results: Vec<RowOut<'a>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stats: Option<StatsOut>,
}

/// The settings of a collection, as [`Create`] takes them.
#[derive(Serialize)]
pub struct ConfigOut<'a> {
    dim: usize,
    metric: &'static str,
    shards: usize,
    fields: FieldsOut<'a>,
    seal_bytes: u64,
    index_bytes: u64,
    compact_deleted: f32,
    max_indexed_segments: usize,
    index: IndexOut,
}

struct FieldsOut<'a>(&'a [Field]);

impl Serialize for FieldsOut<'_> {
    fn serialize<S: Serializer>(&self, s: S) -> std::result::Result<S::Ok, S::Error> {
        s.collect_map(self.0.iter().map(|f| {
            let ty = match f.ty {
                FieldType::Int => "int",
                FieldType::Keyword => "keyword",
            };
            (&f.name, ty)
        }))
    }
}

#[derive(Serialize)]
struct IndexOut {
    max_degree: usize,
    build_list: usize,
    alpha: f32,
    passes: usize,
    pq_m: usize,
    codec: &'static str,
    layout: &'static str,
}

impl<'a> ConfigOut<'a> {
    pub fn new(c: &'a CollectionConfig, shards: usize) -> Self {
        Self {
            dim: c.dim,
            metric: c.metric.name(),
            shards,
            fields: FieldsOut(&c.fields),
            seal_bytes: c.seal_bytes,
            index_bytes: c.index_bytes,
            compact_deleted: c.compact_deleted,
            max_indexed_segments: c.max_indexed_segments,
            index: IndexOut {
                max_degree: c.index.max_degree,
                build_list: c.index.build_list,
                alpha: c.index.alpha,
                passes: c.index.passes,
                pq_m: c.index.pq_m,
                codec: c.index.codec.name(),
                layout: c.index.layout.map_or("auto", Layout::name),
            },
        }
    }
}

#[derive(Serialize)]
pub struct CreatedOut<'a> {
    pub name: &'a str,
    pub config: ConfigOut<'a>,
}

#[derive(Serialize)]
pub struct InfoOut<'a> {
    pub name: &'a str,
    pub config: ConfigOut<'a>,
    pub rows: usize,
    pub state: StateOut,
}

#[derive(Serialize)]
pub struct StateOut {
    /// In RAM and the logs only.
    pub unflushed_rows: usize,
    pub flat_segments: usize,
    pub flat_rows: usize,
    pub ssd_segments: usize,
    pub ssd_rows: usize,
    /// Rows of those segments that are deleted.
    pub deleted_rows: usize,
    pub wal_bytes: u64,
    pub flush_error: Option<String>,
    pub build_error: Option<String>,
    pub optimizing: bool,
    pub optimize_error: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse<'a, T: Deserialize<'a>>(s: &'a str) -> std::result::Result<T, String> {
        serde_json::from_str(s).map_err(|e| e.to_string())
    }

    #[test]
    fn vectors() {
        assert_eq!(parse::<Vector>("[1, -2.5, 1e-3, 16777217]").unwrap().0, [1.0, -2.5, 0.001, 16777216.0]);
        assert!(parse::<Vector>("[]").unwrap().0.is_empty());
        assert_eq!(parse::<Vector>("\"AACAPwAAIMA=\"").unwrap().0, [1.0, -2.5]);
        assert_eq!(parse::<Vector>("\"AACAPwAAIMA\"").unwrap().0, [1.0, -2.5]);
        for bad in ["\"AACAPwAAIM\"", "\"AACAPw!\"", "7", "null", "[\"1\"]", "[null]", "[[1]]", "{}", "[1", "[NaN]"] {
            assert!(parse::<Vector>(bad).is_err(), "{bad}");
        }
        // Too large for an f32: an infinity, which the engine refuses.
        assert_eq!(parse::<Vector>("[1e39]").unwrap().0, [f32::INFINITY]);
    }

    /// Every vector that is sent comes back as the same bits.
    #[test]
    fn floats_survive() {
        let mut rng = vecdb_core::rng::SplitMix64::new(3);
        let random = (0..200_000).map(|_| f32::from_bits(rng.next_u64() as u32));
        let edges = [0.0, -0.0, f32::MIN_POSITIVE, f32::MAX, f32::MIN, f32::EPSILON, 1e-45, 0.1, 16777216.0, 3.0e38];
        let v: Vec<f32> = random.chain(edges).filter(|x| x.is_finite()).collect();
        let text = serde_json::to_string(&v).unwrap();
        let back = parse::<Vector>(&text).unwrap().0;
        assert_eq!(v.len(), back.len());
        for (a, b) in v.iter().zip(&back) {
            assert_eq!(a.to_bits(), b.to_bits(), "{a}");
        }
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        let back = parse::<Vector>(&format!("\"{}\"", base64::encode(&bytes))).unwrap().0;
        assert!(v.iter().zip(&back).all(|(a, b)| a.to_bits() == b.to_bits()) && v.len() == back.len());
    }

    #[test]
    fn rows_and_attributes() {
        let r: RowIn = parse(r#"{"id": 18446744073709551615, "vector": [1], "attrs":
            {"a": -9223372036854775808, "t": ["x", "", "é"], "b": 9223372036854775807, "none": []}}"#)
        .unwrap();
        let r = r.row();
        let want = Attrs::new().int("a", i64::MIN).keyword("t", "x").keyword("t", "").keyword("t", "é").int("b", i64::MAX);
        assert_eq!((r.id, &r.attrs), (u64::MAX, &want));
        let out = serde_json::to_string(&RowOut::new(r.id, Some(0.5), Some(&r), Fetch::ALL)).unwrap();
        assert_eq!(
            out,
            r#"{"id":18446744073709551615,"distance":0.5,"vector":[1.0],"attrs":{"a":-9223372036854775808,"t":["x","","é"],"b":9223372036854775807}}"#
        );
        let many = |n| parse::<RowIn>(&format!(r#"{{"id": 1, "vector": [], "attrs": {{"a": {:?}}}}}"#, vec![7; n]));
        assert!(many(MAX_VALUES).is_ok() && many(MAX_VALUES + 1).is_err());
        let bare = serde_json::to_string(&RowOut::new(7, None, None, Fetch::NONE)).unwrap();
        assert_eq!(bare, r#"{"id":7}"#);
        for bad in [
            r#"{"id": -1, "vector": [1]}"#,
            r#"{"id": 1.5, "vector": [1]}"#,
            r#"{"id": "1", "vector": [1]}"#,
            r#"{"id": 18446744073709551616, "vector": [1]}"#,
            r#"{"vector": [1]}"#,
            r#"{"id": 1}"#,
            r#"{"id": 1, "vector": [1], "attr": {}}"#,
            r#"{"id": 1, "id": 2, "vector": [1]}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": 1.5}}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": true}}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": null}}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": [[1]]}}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": 9223372036854775808}}"#,
            r#"{"id": 1, "vector": [1], "attrs": {"a": 1, "a": 2}}"#,
            r#"{"id": 1, "vector": [1], "attrs": []}"#,
        ] {
            assert!(parse::<RowIn>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn writes() {
        let w: Write = parse(r#"{"ops": [{"upsert": {"id": 1, "vector": [2]}}, {"delete": {"id": 3}}]}"#).unwrap();
        let row = Row { id: 1, vector: vec![2.0], attrs: Attrs::new() };
        assert_eq!(w.ops(), [Op::Upsert(row), Op::Delete { id: 3 }]);
        assert_eq!(parse::<Ids>(r#"{"ids": [4, 5]}"#).unwrap().deletes(), [Op::Delete { id: 4 }, Op::Delete { id: 5 }]);
        for bad in [
            r#"{"ops": [{"upsert": {"id": 1, "vector": [2]}, "delete": {"id": 3}}]}"#,
            r#"{"ops": [{"insert": {"id": 1, "vector": [2]}}]}"#,
            r#"{"ops": [{"delete": 3}]}"#,
            r#"{"ops": [{"delete": {"id": 3, "x": 1}}]}"#,
            r#"{"ops": [], "more": 1}"#,
            r#"{}"#,
        ] {
            assert!(parse::<Write>(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn filters() {
        let f = |s: &str| parse::<FilterIn>(s).map(|f| f.0);
        assert_eq!(f(r#"{"field": "a", "eq": 3}"#).unwrap(), Filter::int_eq("a", 3));
        assert_eq!(f(r#"{"field": "a", "eq": "x"}"#).unwrap(), Filter::keyword_eq("a", "x"));
        assert_eq!(f(r#"{"field": "a", "in": ["x", "y"]}"#).unwrap(), Filter::keyword_in("a", &["x", "y"]));
        assert_eq!(f(r#"{"field": "a", "in": []}"#).unwrap(), Filter::In("a".into(), vec![]));
        assert_eq!(f(r#"{"field": "a", "gte": 1, "lt": 9}"#).unwrap(), Filter::range("a", 1..9));
        assert_eq!(f(r#"{"field": "a", "gt": 1, "lte": 9}"#).unwrap(), Filter::range("a", 2..=9));
        assert_eq!(f(r#"{"field": "a", "lt": 9}"#).unwrap(), Filter::range("a", ..9));
        assert_eq!(f(r#"{"field": "a", "gte": -3}"#).unwrap(), Filter::range("a", -3..));
        // Nothing is above the largest integer.
        let none = f(&format!(r#"{{"field": "a", "gt": {}}}"#, i64::MAX)).unwrap();
        assert!(!none.matches(&Attrs::new().int("a", i64::MAX)));
        assert_eq!(
            f(r#"{"and": [{"not": {"field": "a", "eq": 1}}, {"or": []}]}"#).unwrap(),
            Filter::and([Filter::not(Filter::int_eq("a", 1)), Filter::or([])])
        );
        for bad in [
            r#"{}"#,
            r#"{"field": "a"}"#,
            r#"{"eq": 3}"#,
            r#"{"field": "a", "eq": 3, "in": [4]}"#,
            r#"{"field": "a", "eq": 3, "gt": 4}"#,
            r#"{"field": "a", "gt": 3, "gte": 4}"#,
            r#"{"field": "a", "lt": 3, "lte": 4}"#,
            r#"{"field": "a", "gt": "x"}"#,
            r#"{"field": "a", "gt": 1.5}"#,
            r#"{"field": "a", "eq": null}"#,
            r#"{"field": "a", "eq": [1]}"#,
            r#"{"field": "a", "equals": 3}"#,
            r#"{"and": [], "or": []}"#,
            r#"{"and": [], "field": "a"}"#,
            r#"{"and": {"field": "a", "eq": 1}}"#,
            r#"{"not": []}"#,
            r#"{"and": [{}]}"#,
            r#"[]"#,
        ] {
            assert!(f(bad).is_err(), "{bad}");
        }
        // A filter too deep for the engine, and one too deep to parse, are
        // errors all the same.
        for depth in [40, 100_000] {
            let s = format!(r#"{}{{"field":"a","eq":1}}{}"#, r#"{"not":"#.repeat(depth), "}".repeat(depth));
            if let Ok(f) = f(&s) {
                assert!(depth < 100, "{depth}");
                drop(f);
            }
        }
    }

    #[test]
    fn searches() {
        let req = |s: &str| parse::<Search>(s).and_then(|r| r.request(100, 500, 1000).map_err(|e| e.to_string()));
        let (r, stats) = req(r#"{"vector": [1, 2]}"#).unwrap();
        let d = SearchOptions::default();
        assert_eq!((r.vector, r.k, r.options.l, r.options.beam_width), (vec![1.0, 2.0], 10, d.l, d.beam_width));
        assert_eq!((r.fetch, stats, r.options.filter, r.max_row_bytes), (Fetch::NONE, false, None, 1000));
        let (r, stats) = req(
            r#"{"vector": [1], "k": 100, "l": 500, "beam_width": 8, "rerank": 50, "filter_mode": "scan",
                "filter": {"field": "a", "eq": 1}, "with_attrs": true, "with_vectors": true, "with_stats": true}"#,
        )
        .unwrap();
        assert_eq!((r.k, r.options.l, r.options.beam_width, r.options.rerank), (100, 500, 8, 50));
        assert_eq!((r.options.filter_mode, r.fetch, stats), (FilterMode::Scan, Fetch::ALL, true));
        assert_eq!(r.options.filter, Some(Filter::int_eq("a", 1)));
        for bad in [
            r#""k": 0"#,
            r#""k": 101"#,
            r#""k": 18446744073709551615"#,
            r#""k": -1"#,
            r#""k": 1.0"#,
            r#""l": 0"#,
            r#""l": 501"#,
            r#""rerank": 501"#,
            r#""beam_width": 0"#,
            r#""beam_width": 100000"#,
            r#""filter_mode": "fast""#,
            r#""filter": {}"#,
            r#""with_attr": true"#,
            r#""with_attrs": 1"#,
            r#""k": 1, "k": 2"#,
            r#""vector": [2]"#,
        ] {
            assert!(req(&format!(r#"{{"vector": [1], {bad}}}"#)).is_err(), "{bad}");
        }
        assert!(req(r#"{"k": 3}"#).is_err());
    }

    #[test]
    fn settings_round_trip() {
        let create = |s: &str| parse::<Create>(s).and_then(|c| c.config().map_err(|e| e.to_string()));
        assert_eq!(create(r#"{"dim": 128, "metric": "l2"}"#).unwrap(), CollectionConfig::new(128, Metric::L2));
        let c = create(
            r#"{"dim": 64, "metric": "cosine", "shards": 3, "fields": {"year": "int", "tag": "keyword"},
                "seal_bytes": 1000000, "index_bytes": 2000000, "compact_deleted": 0.5, "max_indexed_segments": 4,
                "index": {"max_degree": 32, "build_list": 64, "alpha": 1.5, "passes": 2, "pq_m": 16,
                          "codec": "f16", "layout": "separate"}}"#,
        )
        .unwrap();
        assert_eq!(c.fields, [Field::int("year"), Field::keyword("tag")]);
        assert_eq!((c.index.codec, c.index.layout), (Codec::F16, Some(Layout::Separate)));
        // What a collection reports can be given to create another.
        for c in [c, CollectionConfig::new(96, Metric::InnerProduct)] {
            let out = serde_json::to_string(&ConfigOut::new(&c, 3)).unwrap();
            assert_eq!(create(&out).unwrap(), c, "{out}");
        }
        for bad in [
            r#""metric": "l3""#,
            r#""metric": "l2", "fields": {"a": "float"}"#,
            r#""metric": "l2", "fields": {"a": "int", "a": "int"}"#,
            r#""metric": "l2", "seal_bytes": 1e18"#,
            r#""metric": "l2", "seal_bytes": 18446744073709551615"#,
            r#""metric": "l2", "index_bytes": 18446744073709551615"#,
            r#""metric": "l2", "index": {"codec": "f64"}"#,
            r#""metric": "l2", "index": {"layout": "both"}"#,
            r#""metric": "l2", "index": {"build_list": 1000000}"#,
            r#""metric": "l2", "index": {"passes": 1000}"#,
            r#""metric": "l2", "index": {"degree": 3}"#,
            r#""metric": "l2", "dims": 3"#,
        ] {
            assert!(create(&format!(r#"{{"dim": 8, {bad}}}"#)).is_err(), "{bad}");
        }
        assert!(create(r#"{"metric": "l2"}"#).is_err() && create(r#"{"dim": -1, "metric": "l2"}"#).is_err());
    }
}
