//! Rank fusion for combining retrieval results.
//!
//! ```rust
//! use rank_fusion::{rrf, RrfConfig};
//!
//! let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
//! let dense = vec![("d2", 0.9), ("d3", 0.8)];
//! let fused = rrf(bm25, dense, RrfConfig::default());
//! ```
//!
//! ## Two-List Functions
//!
//! - [`rrf`] — Reciprocal Rank Fusion (ignores scores, uses rank)
//! - [`combsum`] — Sum of normalized scores
//! - [`combmnz`] — Sum × overlap count
//! - [`borda`] — Borda count
//! - [`weighted`] — Weighted combination
//!
//! ## Multi-List Functions
//!
//! - [`rrf_multi`], [`combsum_multi`], [`combmnz_multi`], [`borda_multi`], [`weighted_multi`]
//!
//! ## Choosing an Algorithm
//!
//! | Scenario | Recommended |
//! |----------|-------------|
//! | Different score scales (BM25 + cosine) | `rrf` — ignores scores |
//! | Same score scale, reward overlap | `combmnz` |
//! | Trust one retriever more | `weighted` |
//! | Simple voting | `borda` |

use std::collections::HashMap;
use std::hash::Hash;

// ─────────────────────────────────────────────────────────────────────────────
// Error Types
// ─────────────────────────────────────────────────────────────────────────────

/// Errors that can occur during fusion.
#[derive(Debug, Clone, PartialEq)]
pub enum FusionError {
    /// Weights sum to zero or near-zero.
    ZeroWeights,
    /// Invalid configuration parameter.
    InvalidConfig(String),
}

impl std::fmt::Display for FusionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::ZeroWeights => write!(f, "weights sum to zero"),
            Self::InvalidConfig(msg) => write!(f, "invalid config: {msg}"),
        }
    }
}

impl std::error::Error for FusionError {}

/// Result type for fusion operations.
pub type Result<T> = std::result::Result<T, FusionError>;

// ─────────────────────────────────────────────────────────────────────────────
// Configuration with Builder Pattern
// ─────────────────────────────────────────────────────────────────────────────

/// RRF configuration.
///
/// # Example
///
/// ```rust
/// use rank_fusion::RrfConfig;
///
/// let config = RrfConfig::default()
///     .with_k(60)
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RrfConfig {
    /// Smoothing constant (default: 60).
    pub k: u32,
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for RrfConfig {
    fn default() -> Self {
        Self { k: 60, top_k: None }
    }
}

impl RrfConfig {
    /// Create config with custom k.
    #[must_use]
    pub const fn new(k: u32) -> Self {
        Self { k, top_k: None }
    }

    /// Set the k parameter (smoothing constant).
    ///
    /// - `k=60` — Standard RRF, works well for most cases
    /// - `k=1` — Top positions dominate heavily
    /// - `k=100+` — More uniform contribution across ranks
    #[must_use]
    pub const fn with_k(mut self, k: u32) -> Self {
        self.k = k;
        self
    }

    /// Limit output to top_k results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Weighted fusion configuration.
///
/// # Example
///
/// ```rust
/// use rank_fusion::WeightedConfig;
///
/// let config = WeightedConfig::default()
///     .with_weights(0.7, 0.3)
///     .with_normalize(true)
///     .with_top_k(10);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WeightedConfig {
    /// Weight for first list (default: 0.5).
    pub weight_a: f32,
    /// Weight for second list (default: 0.5).
    pub weight_b: f32,
    /// Normalize scores to [0,1] before combining (default: true).
    pub normalize: bool,
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl Default for WeightedConfig {
    fn default() -> Self {
        Self {
            weight_a: 0.5,
            weight_b: 0.5,
            normalize: true,
            top_k: None,
        }
    }
}

impl WeightedConfig {
    /// Create config with custom weights.
    #[must_use]
    pub const fn new(weight_a: f32, weight_b: f32) -> Self {
        Self {
            weight_a,
            weight_b,
            normalize: true,
            top_k: None,
        }
    }

    /// Set weights for the two lists.
    #[must_use]
    pub const fn with_weights(mut self, weight_a: f32, weight_b: f32) -> Self {
        self.weight_a = weight_a;
        self.weight_b = weight_b;
        self
    }

    /// Enable/disable score normalization.
    #[must_use]
    pub const fn with_normalize(mut self, normalize: bool) -> Self {
        self.normalize = normalize;
        self
    }

    /// Limit output to top_k results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Configuration for rank-based fusion (Borda, CombSUM, CombMNZ).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusionConfig {
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl FusionConfig {
    /// Limit output to top_k results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Newtype for Type Safety
// ─────────────────────────────────────────────────────────────────────────────

/// A ranked result list. Newtype for clarity in function signatures.
///
/// The inner slice contains `(id, score)` pairs, sorted by descending score.
#[derive(Debug, Clone, Copy)]
pub struct RankedList<'a, I>(pub &'a [(I, f32)]);

impl<'a, I> AsRef<[(I, f32)]> for RankedList<'a, I> {
    fn as_ref(&self) -> &[(I, f32)] {
        self.0
    }
}

impl<'a, I> From<&'a [(I, f32)]> for RankedList<'a, I> {
    fn from(slice: &'a [(I, f32)]) -> Self {
        Self(slice)
    }
}

impl<'a, I> From<&'a Vec<(I, f32)>> for RankedList<'a, I> {
    fn from(vec: &'a Vec<(I, f32)>) -> Self {
        Self(vec.as_slice())
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// RRF (Reciprocal Rank Fusion)
// ─────────────────────────────────────────────────────────────────────────────

/// Reciprocal Rank Fusion of two result lists.
///
/// Formula: `score(d) = Σ 1/(k + rank)` where rank is 0-indexed.
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

    finalize(scores, config.top_k)
}

/// RRF with preallocated output buffer.
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
    if let Some(top_k) = config.top_k {
        output.truncate(top_k);
    }
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

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Score-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted score fusion with optional normalization.
///
/// Formula: `score(d) = w_a × norm(s_a) + w_b × norm(s_b)`
pub fn weighted<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: WeightedConfig,
) -> Vec<(I, f32)> {
    weighted_impl(
        &[(results_a, config.weight_a), (results_b, config.weight_b)],
        config.normalize,
        config.top_k,
    )
}

/// Weighted fusion for 3+ result lists.
///
/// Each list is paired with its weight. Weights are normalized to sum to 1.
///
/// # Errors
///
/// Returns `Err(FusionError::ZeroWeights)` if weights sum to zero.
pub fn weighted_multi<I, L>(
    lists: &[(L, f32)],
    normalize: bool,
    top_k: Option<usize>,
) -> Result<Vec<(I, f32)>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < 1e-9 {
        return Err(FusionError::ZeroWeights);
    }

    let mut scores: HashMap<I, f32> = HashMap::new();

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            *scores.entry(id.clone()).or_default() += w * (s - off) * norm;
        }
    }

    Ok(finalize(scores, top_k))
}

/// Internal weighted implementation (infallible for two-list case).
fn weighted_impl<I, L>(lists: &[(L, f32)], normalize: bool, top_k: Option<usize>) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < 1e-9 {
        return Vec::new();
    }

    let mut scores: HashMap<I, f32> = HashMap::new();

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            *scores.entry(id.clone()).or_default() += w * (s - off) * norm;
        }
    }

    finalize(scores, top_k)
}

/// `CombSUM` — sum of normalized scores.
pub fn combsum<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combsum_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombSUM` with configuration.
pub fn combsum_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combsum_multi(&[results_a, results_b], config)
}

/// `CombSUM` for 3+ result lists.
pub fn combsum_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            *scores.entry(id.clone()).or_default() += (s - off) * norm;
        }
    }

    finalize(scores, config.top_k)
}

/// `CombMNZ` — sum × number of lists containing the document.
pub fn combmnz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmnz_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombMNZ` with configuration.
pub fn combmnz_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combmnz_multi(&[results_a, results_b], config)
}

/// `CombMNZ` for 3+ result lists.
#[allow(clippy::cast_precision_loss)]
pub fn combmnz_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let mut scores: HashMap<I, (f32, u32)> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            let e = scores.entry(id.clone()).or_default();
            e.0 += (s - off) * norm;
            e.1 += 1;
        }
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (sum, n))| (id, sum * n as f32))
        .collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Borda count — each position contributes `N - rank` points.
pub fn borda<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    borda_with_config(results_a, results_b, FusionConfig::default())
}

/// Borda count with configuration.
pub fn borda_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    borda_multi(&[results_a, results_b], config)
}

/// Borda count for 3+ result lists.
#[allow(clippy::cast_precision_loss)]
pub fn borda_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let n = items.len() as f32;
        for (rank, (id, _)) in items.iter().enumerate() {
            *scores.entry(id.clone()).or_default() += n - rank as f32;
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sort scores descending and optionally truncate.
#[inline]
fn finalize<I>(scores: HashMap<I, f32>, top_k: Option<usize>) -> Vec<(I, f32)> {
    let mut results: Vec<_> = scores.into_iter().collect();
    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(k) = top_k {
        results.truncate(k);
    }
    results
}

/// Returns (scale, offset) for min-max normalization: `(x - offset) * scale`.
#[inline]
fn min_max_params<I>(results: &[(I, f32)]) -> (f32, f32) {
    if results.is_empty() {
        return (1.0, 0.0);
    }
    let (min, max) = results
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), (_, s)| {
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
    fn rrf_with_top_k() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = rrf(a, b, RrfConfig::default().with_top_k(2));

        assert_eq!(f.len(), 2);
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
        let a = vec![("d1", 1.0)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf(a, b, RrfConfig::new(60));

        let expected = 1.0 / 60.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn combmnz_rewards_overlap() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let f = combmnz(&a, &b);

        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn combsum_basic() {
        let a = vec![("d1", 0.5), ("d2", 1.0)];
        let b = vec![("d2", 1.0), ("d3", 0.5)];
        let f = combsum(&a, &b);

        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn weighted_skewed() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d2", 1.0)];

        let f = weighted(
            &a,
            &b,
            WeightedConfig::default()
                .with_weights(0.9, 0.1)
                .with_normalize(false),
        );
        assert_eq!(f[0].0, "d1");

        let f = weighted(
            &a,
            &b,
            WeightedConfig::default()
                .with_weights(0.1, 0.9)
                .with_normalize(false),
        );
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn borda_symmetric() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);
        let f = borda(&a, &b);

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
    fn borda_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];
        let f = borda_multi(&lists, FusionConfig::default());
        assert_eq!(f.len(), 3);
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn combsum_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            vec![("d1", 1.0), ("d2", 0.5)],
            vec![("d2", 1.0), ("d3", 0.5)],
            vec![("d1", 1.0), ("d3", 0.5)],
        ];
        let f = combsum_multi(&lists, FusionConfig::default());
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn combmnz_multi_works() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            vec![("d1", 1.0)],
            vec![("d1", 1.0), ("d2", 0.5)],
            vec![("d1", 1.0), ("d2", 0.5)],
        ];
        let f = combmnz_multi(&lists, FusionConfig::default());
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn weighted_multi_works() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d2", 1.0)];
        let c = vec![("d3", 1.0)];

        let f = weighted_multi(&[(&a, 1.0), (&b, 1.0), (&c, 1.0)], false, None).unwrap();
        assert_eq!(f.len(), 3);

        let f = weighted_multi(&[(&a, 10.0), (&b, 1.0), (&c, 1.0)], false, None).unwrap();
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn weighted_multi_zero_weights() {
        let a = vec![("d1", 1.0)];
        let result = weighted_multi(&[(&a, 0.0)], false, None);
        assert!(matches!(result, Err(FusionError::ZeroWeights)));
    }

    #[test]
    fn empty_inputs() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = ranked(&["d1"]);

        assert_eq!(
            rrf(empty.clone(), non_empty.clone(), RrfConfig::default()).len(),
            1
        );
        assert_eq!(rrf(non_empty, empty, RrfConfig::default()).len(), 1);
    }

    #[test]
    fn both_empty() {
        let empty: Vec<(&str, f32)> = vec![];
        assert_eq!(
            rrf(empty.clone(), empty.clone(), RrfConfig::default()).len(),
            0
        );
        assert_eq!(combsum(&empty, &empty).len(), 0);
        assert_eq!(borda(&empty, &empty).len(), 0);
    }

    #[test]
    fn duplicate_ids_in_same_list() {
        let a = vec![("d1", 1.0), ("d1", 0.5)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf(a, b, RrfConfig::new(60));

        assert_eq!(f.len(), 1);
        let expected = 1.0 / 60.0 + 1.0 / 61.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn builder_pattern() {
        let config = RrfConfig::default().with_k(30).with_top_k(5);
        assert_eq!(config.k, 30);
        assert_eq!(config.top_k, Some(5));

        let config = WeightedConfig::default()
            .with_weights(0.8, 0.2)
            .with_normalize(false)
            .with_top_k(10);
        assert_eq!(config.weight_a, 0.8);
        assert!(!config.normalize);
        assert_eq!(config.top_k, Some(10));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Property Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;

    fn arb_results(max_len: usize) -> impl Strategy<Value = Vec<(u32, f32)>> {
        proptest::collection::vec((0u32..100, 0.0f32..1.0), 0..max_len)
    }

    proptest! {
        #[test]
        fn rrf_output_bounded(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(a.clone(), b.clone(), RrfConfig::default());
            prop_assert!(result.len() <= a.len() + b.len());
        }

        #[test]
        fn rrf_scores_positive(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(a, b, RrfConfig::default());
            for (_, score) in &result {
                prop_assert!(*score > 0.0);
            }
        }

        #[test]
        fn rrf_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = rrf(a.clone(), b.clone(), RrfConfig::default());
            let ba = rrf(b, a, RrfConfig::default());

            prop_assert_eq!(ab.len(), ba.len());

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6);
            }
        }

        #[test]
        fn rrf_sorted_descending(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(a, b, RrfConfig::default());
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        #[test]
        fn rrf_top_k_respected(a in arb_results(50), b in arb_results(50), k in 1usize..20) {
            let result = rrf(a, b, RrfConfig::default().with_top_k(k));
            prop_assert!(result.len() <= k);
        }

        #[test]
        fn borda_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = borda(&a, &b);
            let ba = borda(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();
            prop_assert_eq!(ab_map, ba_map);
        }

        #[test]
        fn combsum_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combsum(&a, &b);
            let ba = combsum(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).unwrap();
                prop_assert!((score_ab - score_ba).abs() < 1e-5);
            }
        }

        #[test]
        fn rrf_k_uniformity(a in arb_results(10).prop_filter("need items", |v| v.len() >= 2)) {
            let b: Vec<(u32, f32)> = vec![];

            let low_k = rrf(a.clone(), b.clone(), RrfConfig::new(1));
            let high_k = rrf(a, b, RrfConfig::new(1000));

            if low_k.len() >= 2 && high_k.len() >= 2 {
                let low_k_range = low_k[0].1 - low_k[low_k.len()-1].1;
                let high_k_range = high_k[0].1 - high_k[high_k.len()-1].1;
                prop_assert!(high_k_range <= low_k_range);
            }
        }
    }
}

