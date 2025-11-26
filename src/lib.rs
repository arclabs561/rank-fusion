//! # rank-fusion
//!
//! Rank fusion algorithms for hybrid search systems.
//!
//! This crate provides implementations of common rank fusion algorithms used
//! to combine results from multiple retrieval systems (e.g., BM25 + dense
//! vectors in RAG applications).
//!
//! ## Algorithms
//!
//! - **RRF** — Reciprocal Rank Fusion, robust to score distribution differences
//! - **CombSUM** / **CombMNZ** — Score-based combination with overlap rewards
//! - **Borda** — Rank-based voting
//! - **Weighted** — Configurable score weighting with normalization
//!
//! ## Example
//!
//! ```rust
//! use rank_fusion::{rrf, RrfConfig};
//!
//! let sparse = vec![("doc1", 0.9), ("doc2", 0.7), ("doc3", 0.5)];
//! let dense = vec![("doc2", 0.85), ("doc4", 0.6), ("doc1", 0.4)];
//!
//! let fused = rrf(sparse, dense, RrfConfig::default());
//! assert_eq!(fused[0].0, "doc2"); // appears in both lists
//! ```

use std::collections::HashMap;
use std::hash::Hash;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration
// ─────────────────────────────────────────────────────────────────────────────

/// RRF configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrfConfig {
    /// Smoothing constant (default: 60).
    ///
    /// Controls how much rank position affects the score:
    /// - `k=60` — Standard RRF, works well for most cases
    /// - `k=1` — Top positions dominate heavily
    /// - `k=100+` — More uniform contribution across ranks
    pub k: u32,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60 }
    }
}

impl RrfConfig {
    /// Create config with custom k.
    #[must_use]
    pub const fn new(k: u32) -> Self {
        Self { k }
    }
}

/// Weighted fusion configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedConfig {
    /// Weight for first list (default: 0.5).
    pub weight_a: f32,
    /// Weight for second list (default: 0.5).
    pub weight_b: f32,
    /// Normalize scores to [0,1] before combining (default: true).
    pub normalize: bool,
}

impl Default for WeightedConfig {
    fn default() -> Self {
        Self {
            weight_a: 0.5,
            weight_b: 0.5,
            normalize: true,
        }
    }
}

impl WeightedConfig {
    /// Create config with custom weights (normalized internally).
    #[must_use]
    pub const fn new(weight_a: f32, weight_b: f32) -> Self {
        Self {
            weight_a,
            weight_b,
            normalize: true,
        }
    }

    /// Create config with custom weights and normalization setting.
    #[must_use]
    pub const fn with_normalize(weight_a: f32, weight_b: f32, normalize: bool) -> Self {
        Self {
            weight_a,
            weight_b,
            normalize,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RRF (Reciprocal Rank Fusion)
// ─────────────────────────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion of two result lists.
///
/// Formula: `score(d) = Σ 1/(k + rank)` where rank is 0-indexed.
///
/// For the first item (rank=0) with k=60, contribution is 1/60 ≈ 0.0167.
///
/// RRF is robust to score distribution differences between retrievers.
/// The original scores are ignored; only rank position matters.
///
/// # Example
///
/// ```rust
/// use rank_fusion::{rrf, RrfConfig};
///
/// let sparse = vec![("d1", 0.9), ("d2", 0.5)];
/// let dense = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = rrf(sparse, dense, RrfConfig::default());
/// assert_eq!(fused[0].0, "d2"); // appears in both
/// ```
#[allow(clippy::cast_precision_loss)]
pub fn rrf<I, A, B>(results_a: A, results_b: B, config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    A: IntoIterator<Item = (I, f32)>,
    B: IntoIterator<Item = (I, f32)>,
{
    let k = config.k as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();

    for (rank, (id, _)) in results_a.into_iter().enumerate() {
        *scores.entry(id).or_default() += 1.0 / (k + rank as f32);
    }
    for (rank, (id, _)) in results_b.into_iter().enumerate() {
        *scores.entry(id).or_default() += 1.0 / (k + rank as f32);
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// RRF with preallocated output buffer.
///
/// Reuses the output Vec but still allocates a HashMap internally for
/// score accumulation.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_into<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
    output: &mut Vec<(I, f32)>,
) {
    output.clear();
    let k = config.k as f32;
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(results_a.len() + results_b.len());

    for (rank, (id, _)) in results_a.iter().enumerate() {
        *scores.entry(id.clone()).or_default() += 1.0 / (k + rank as f32);
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        *scores.entry(id.clone()).or_default() += 1.0 / (k + rank as f32);
    }

    output.extend(scores);
    output.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
}

/// RRF for 3+ result lists.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_multi<I, L>(lists: &[L], config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let k = config.k as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            *scores.entry(id.clone()).or_default() += 1.0 / (k + rank as f32);
        }
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Score-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted score fusion with optional normalization.
///
/// Formula: `score(d) = w_a × norm(s_a) + w_b × norm(s_b)`
///
/// When `normalize` is true, scores are scaled to [0,1] using min-max
/// normalization before weighting.
pub fn weighted<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: WeightedConfig,
) -> Vec<(I, f32)> {
    let (norm_a, off_a) = if config.normalize { min_max_params(results_a) } else { (1.0, 0.0) };
    let (norm_b, off_b) = if config.normalize { min_max_params(results_b) } else { (1.0, 0.0) };

    let total = config.weight_a + config.weight_b;
    let wa = config.weight_a / total;
    let wb = config.weight_b / total;

    let mut scores: HashMap<I, f32> = HashMap::new();

    for (id, s) in results_a {
        *scores.entry(id.clone()).or_default() += wa * (s - off_a) * norm_a;
    }
    for (id, s) in results_b {
        *scores.entry(id.clone()).or_default() += wb * (s - off_b) * norm_b;
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

/// `CombSUM` — sum of normalized scores.
pub fn combsum<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    weighted(results_a, results_b, WeightedConfig::default())
}

/// `CombMNZ` — sum × number of lists containing the document.
///
/// Rewards documents appearing in multiple lists.
#[allow(clippy::cast_precision_loss)]
pub fn combmnz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    let (norm_a, off_a) = min_max_params(results_a);
    let (norm_b, off_b) = min_max_params(results_b);

    let mut scores: HashMap<I, (f32, u32)> = HashMap::new();

    for (id, s) in results_a {
        let e = scores.entry(id.clone()).or_default();
        e.0 += (s - off_a) * norm_a;
        e.1 += 1;
    }
    for (id, s) in results_b {
        let e = scores.entry(id.clone()).or_default();
        e.0 += (s - off_b) * norm_b;
        e.1 += 1;
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (sum, n))| (id, sum * n as f32))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Borda count — each position contributes `N - rank` points.
///
/// For a list of N items, position 0 gets N points, position 1 gets N-1, etc.
#[allow(clippy::cast_precision_loss)]
pub fn borda<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    let n_a = results_a.len() as f32;
    let n_b = results_b.len() as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();

    for (rank, (id, _)) in results_a.iter().enumerate() {
        *scores.entry(id.clone()).or_default() += n_a - rank as f32;
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        *scores.entry(id.clone()).or_default() += n_b - rank as f32;
    }

    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Returns (scale, offset) for min-max normalization: `(x - offset) * scale`.
fn min_max_params<I>(results: &[(I, f32)]) -> (f32, f32) {
    if results.is_empty() {
        return (1.0, 0.0);
    }
    let (min, max) = results.iter().fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, s)| {
        (lo.min(*s), hi.max(*s))
    });
    let range = max - min;
    if range < 1e-9 {
        (1.0, min)
    } else {
        (1.0 / range, min)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn ranked<'a>(ids: &[&'a str]) -> Vec<(&'a str, f32)> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| (id, 1.0 - i as f32 * 0.1))
            .collect()
    }

    #[test]
    fn rrf_basic() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = rrf(a, b, RrfConfig::default());

        assert!(f.iter().position(|(id, _)| *id == "d2").unwrap() < 2);
    }

    #[test]
    fn rrf_into_works() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let mut out = Vec::new();

        rrf_into(&a, &b, RrfConfig::default(), &mut out);

        assert_eq!(out.len(), 3);
        assert_eq!(out[0].0, "d2");
    }

    #[test]
    fn rrf_score_formula() {
        // Verify the RRF formula: 1/(k + rank) where rank is 0-indexed
        let a = vec![("d1", 1.0)]; // rank 0
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf(a, b, RrfConfig::new(60));

        // First item (rank=0) should get 1/60
        let expected = 1.0 / 60.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn combmnz_rewards_overlap() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let f = combmnz(&a, &b);

        assert_eq!(f[0].0, "d2"); // 2× multiplier
    }

    #[test]
    fn combsum_basic() {
        let a = vec![("d1", 1.0), ("d2", 0.5)];
        let b = vec![("d2", 1.0), ("d3", 0.5)];
        let f = combsum(&a, &b);

        // d2 appears in both with high scores
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn weighted_skewed() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d2", 1.0)];

        // 90% weight to list a
        let f = weighted(&a, &b, WeightedConfig::with_normalize(0.9, 0.1, false));
        assert_eq!(f[0].0, "d1");

        // 90% weight to list b
        let f = weighted(&a, &b, WeightedConfig::with_normalize(0.1, 0.9, false));
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn borda_symmetric() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);
        let f = borda(&a, &b);

        // All equal: d1=3+1=4, d2=2+2=4, d3=1+3=4
        let scores: Vec<f32> = f.iter().map(|(_, s)| *s).collect();
        assert!((scores[0] - scores[1]).abs() < 0.01);
        assert!((scores[1] - scores[2]).abs() < 0.01);
    }

    #[test]
    fn rrf_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];
        let f = rrf_multi(&lists, RrfConfig::default());

        assert_eq!(f.len(), 3);
    }

    #[test]
    fn empty_inputs() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = ranked(&["d1"]);

        assert_eq!(rrf(empty.clone(), non_empty.clone(), RrfConfig::default()).len(), 1);
        assert_eq!(rrf(non_empty, empty, RrfConfig::default()).len(), 1);
    }

    #[test]
    fn both_empty() {
        let empty: Vec<(&str, f32)> = vec![];
        assert_eq!(rrf(empty.clone(), empty.clone(), RrfConfig::default()).len(), 0);
        assert_eq!(combsum(&empty, &empty).len(), 0);
        assert_eq!(borda(&empty, &empty).len(), 0);
    }

    #[test]
    fn duplicate_ids_in_same_list() {
        // If same doc appears twice in one list, it should accumulate
        let a = vec![("d1", 1.0), ("d1", 0.5)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf(a, b, RrfConfig::new(60));

        assert_eq!(f.len(), 1);
        // Should get 1/60 + 1/61
        let expected = 1.0 / 60.0 + 1.0 / 61.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }
}
