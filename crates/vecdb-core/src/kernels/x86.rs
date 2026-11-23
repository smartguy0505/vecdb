//! AVX2 + FMA kernels for x86_64.
//!
//! Soundness: the `#[target_feature]` functions may only run on a CPU with
//! AVX2 and FMA. The safe wrappers at the bottom are reachable only through
//! `AVX2_FMA`, which `avx2_fma()` hands out only after runtime detection.
//! Every raw-pointer load is bounded by the length of *both* slices, so a
//! length mismatch can never read out of bounds; the wrappers additionally
//! assert equal lengths, so a mismatch panics instead of truncating.

#![allow(unsafe_code)]

use std::arch::x86_64::*;

use super::{Kernels, check_batch};

static AVX2_FMA: Kernels = Kernels {
    name: "avx2-fma",
    dot,
    l2_squared,
    dot_batch,
    l2_squared_batch,
};

pub(super) fn avx2_fma() -> Option<&'static Kernels> {
    if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
        Some(&AVX2_FMA)
    } else {
        None
    }
}

/// Prefetch hint for the first `PREFETCH_BYTES` of `data` (SSE, which every
/// x86_64 CPU has, so no runtime detection is needed).
#[inline]
pub(super) fn prefetch<T>(data: &[T]) {
    let bytes = std::mem::size_of_val(data).min(super::PREFETCH_BYTES);
    let p = data.as_ptr().cast::<i8>();
    let mut off = 0;
    while off < bytes {
        // SAFETY: `off < size_of_val(data)`, so the address is inside `data`;
        // a prefetch is only a hint and cannot fault or change memory.
        unsafe { _mm_prefetch::<_MM_HINT_T0>(p.add(off)) };
        off += 64;
    }
}

/// Horizontal sum of 8 lanes.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn hsum(v: __m256) -> f32 {
    let s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps::<1>(v));
    let s = _mm_add_ps(s, _mm_movehl_ps(s, s));
    let s = _mm_add_ss(s, _mm_movehdup_ps(s));
    _mm_cvtss_f32(s)
}

/// Four independent 8-lane accumulators (32 floats per iteration) hide FMA
/// latency; then single 8-lane steps; then a scalar tail of < 8 values.
#[inline]
#[target_feature(enable = "avx2,fma")]
fn dot_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY: every load reads 8 floats starting at i (+0/8/16/24) and the
    // loop conditions keep the last one below n <= a.len(), b.len().
    unsafe {
        while i + 32 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)), acc1);
            acc2 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)), acc2);
            acc3 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)), acc3);
            i += 32;
        }
        while i + 8 <= n {
            acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)), acc0);
            i += 8;
        }
    }
    let mut sum = hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for j in i..n {
        sum += a[j] * b[j];
    }
    sum
}

#[inline]
#[target_feature(enable = "avx2,fma")]
fn l2_avx2(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len().min(b.len());
    let (pa, pb) = (a.as_ptr(), b.as_ptr());
    let mut acc0 = _mm256_setzero_ps();
    let mut acc1 = _mm256_setzero_ps();
    let mut acc2 = _mm256_setzero_ps();
    let mut acc3 = _mm256_setzero_ps();
    let mut i = 0;
    // SAFETY: as in dot_avx2.
    unsafe {
        while i + 32 <= n {
            let d0 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            let d1 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 8)), _mm256_loadu_ps(pb.add(i + 8)));
            let d2 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 16)), _mm256_loadu_ps(pb.add(i + 16)));
            let d3 = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i + 24)), _mm256_loadu_ps(pb.add(i + 24)));
            acc0 = _mm256_fmadd_ps(d0, d0, acc0);
            acc1 = _mm256_fmadd_ps(d1, d1, acc1);
            acc2 = _mm256_fmadd_ps(d2, d2, acc2);
            acc3 = _mm256_fmadd_ps(d3, d3, acc3);
            i += 32;
        }
        while i + 8 <= n {
            let d = _mm256_sub_ps(_mm256_loadu_ps(pa.add(i)), _mm256_loadu_ps(pb.add(i)));
            acc0 = _mm256_fmadd_ps(d, d, acc0);
            i += 8;
        }
    }
    let mut sum = hsum(_mm256_add_ps(_mm256_add_ps(acc0, acc1), _mm256_add_ps(acc2, acc3)));
    for j in i..n {
        let d = a[j] - b[j];
        sum += d * d;
    }
    sum
}

#[target_feature(enable = "avx2,fma")]
fn dot_batch_avx2(query: &[f32], rows: &[f32], out: &mut [f32]) {
    for (row, o) in rows.chunks_exact(query.len()).zip(out.iter_mut()) {
        *o = dot_avx2(query, row);
    }
}

#[target_feature(enable = "avx2,fma")]
fn l2_batch_avx2(query: &[f32], rows: &[f32], out: &mut [f32]) {
    for (row, o) in rows.chunks_exact(query.len()).zip(out.iter_mut()) {
        *o = l2_avx2(query, row);
    }
}

// Safe wrappers, reachable only via AVX2_FMA (see module docs).

fn dot(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "dot: length mismatch");
    // SAFETY: AVX2 and FMA were detected before AVX2_FMA was handed out.
    unsafe { dot_avx2(a, b) }
}

fn l2_squared(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "l2_squared: length mismatch");
    // SAFETY: as in dot.
    unsafe { l2_avx2(a, b) }
}

fn dot_batch(query: &[f32], rows: &[f32], out: &mut [f32]) {
    check_batch(query, rows, out);
    // SAFETY: as in dot.
    unsafe { dot_batch_avx2(query, rows, out) }
}

fn l2_squared_batch(query: &[f32], rows: &[f32], out: &mut [f32]) {
    check_batch(query, rows, out);
    // SAFETY: as in dot.
    unsafe { l2_batch_avx2(query, rows, out) }
}
