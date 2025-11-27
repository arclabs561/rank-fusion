//! Rank fusion for hybrid search.
//!
//! Combine results from multiple retrievers (BM25, dense, sparse) into a single ranking.
//!
//! ```rust
//! use rank_fusion::{rrf, RrfConfig};
//!
//! let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
//! let dense = vec![("d2", 0.9), ("d3", 0.8)];
//! let fused = rrf(bm25, dense, RrfConfig::default());
//! // d2 ranks highest (appears in both lists)
//! ```
//!
//! # Algorithms
//!
//! | Function | Uses Scores | Best For |
//! |----------|-------------|----------|
//! | [`rrf`] | No | Incompatible score scales |
//! | [`combsum`] | Yes | Similar scales, trust scores |
//! | [`combmnz`] | Yes | Reward overlap between lists |
//! | [`borda`] | No | Simple voting |
//! | [`weighted`] | Yes | Custom retriever weights |
//! | [`dbsf`] | Yes | Different score distributions |
//!
//! All have `*_multi` variants for 3+ lists.

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
    /// Normalize scores to `[0,1]` before combining (default: true).
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
// Prelude
// ─────────────────────────────────────────────────────────────────────────────

/// Prelude for common imports.
///
/// ```rust
/// use rank_fusion::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{borda, combmnz, combsum, dbsf, rrf, weighted};
    pub use crate::{FusionConfig, FusionError, FusionMethod, Result, RrfConfig, WeightedConfig};
}

// ─────────────────────────────────────────────────────────────────────────────
// Unified Fusion Method
// ─────────────────────────────────────────────────────────────────────────────

/// Unified fusion method for dispatching to different algorithms.
///
/// Provides a single entry point for all fusion algorithms with a consistent API.
///
/// # Example
///
/// ```rust
/// use rank_fusion::FusionMethod;
///
/// let sparse = vec![("d1", 10.0), ("d2", 8.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.7)];
///
/// // Use RRF (rank-based, score-agnostic)
/// let fused = FusionMethod::Rrf { k: 60 }.fuse(&sparse, &dense);
///
/// // Use CombSUM (score-based)
/// let fused = FusionMethod::CombSum.fuse(&sparse, &dense);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FusionMethod {
    /// Reciprocal Rank Fusion (ignores scores, uses rank position).
    Rrf {
        /// Smoothing constant (default: 60).
        k: u32,
    },
    /// CombSUM — sum of normalized scores.
    CombSum,
    /// CombMNZ — sum × overlap count.
    CombMnz,
    /// Borda count — N - rank points.
    Borda,
    /// Weighted combination with custom weights.
    Weighted {
        /// Weight for first list.
        weight_a: f32,
        /// Weight for second list.
        weight_b: f32,
        /// Whether to normalize scores before combining.
        normalize: bool,
    },
    /// Distribution-Based Score Fusion (z-score normalization).
    Dbsf,
}

impl Default for FusionMethod {
    fn default() -> Self {
        Self::Rrf { k: 60 }
    }
}

impl FusionMethod {
    /// Create RRF method with default k=60.
    #[must_use]
    pub const fn rrf() -> Self {
        Self::Rrf { k: 60 }
    }

    /// Create RRF method with custom k.
    #[must_use]
    pub const fn rrf_with_k(k: u32) -> Self {
        Self::Rrf { k }
    }

    /// Create weighted method with custom weights.
    #[must_use]
    pub const fn weighted(weight_a: f32, weight_b: f32) -> Self {
        Self::Weighted {
            weight_a,
            weight_b,
            normalize: true,
        }
    }

    /// Fuse two ranked lists using this method.
    ///
    /// # Arguments
    /// * `a` - First ranked list (ID, score pairs)
    /// * `b` - Second ranked list (ID, score pairs)
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    #[must_use]
    pub fn fuse<I: Clone + Eq + Hash>(&self, a: &[(I, f32)], b: &[(I, f32)]) -> Vec<(I, f32)> {
        match self {
            Self::Rrf { k } => crate::rrf_multi(&[a, b], RrfConfig::new(*k)),
            Self::CombSum => crate::combsum(a, b),
            Self::CombMnz => crate::combmnz(a, b),
            Self::Borda => crate::borda(a, b),
            Self::Weighted {
                weight_a,
                weight_b,
                normalize,
            } => crate::weighted(
                a,
                b,
                WeightedConfig::new(*weight_a, *weight_b).with_normalize(*normalize),
            ),
            Self::Dbsf => crate::dbsf(a, b),
        }
    }

    /// Fuse multiple ranked lists using this method.
    ///
    /// # Arguments
    /// * `lists` - Slice of ranked lists
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    #[must_use]
    pub fn fuse_multi<I, L>(&self, lists: &[L]) -> Vec<(I, f32)>
    where
        I: Clone + Eq + Hash,
        L: AsRef<[(I, f32)]>,
    {
        match self {
            Self::Rrf { k } => crate::rrf_multi(lists, RrfConfig::new(*k)),
            Self::CombSum => crate::combsum_multi(lists, FusionConfig::default()),
            Self::CombMnz => crate::combmnz_multi(lists, FusionConfig::default()),
            Self::Borda => crate::borda_multi(lists, FusionConfig::default()),
            Self::Weighted { .. } => {
                // For multi-list weighted, use equal weights
                // (users should use weighted_multi directly for custom weights)
                if lists.len() == 2 {
                    self.fuse(lists[0].as_ref(), lists[1].as_ref())
                } else {
                    // Fall back to equal-weighted combsum for 3+ lists
                    crate::combsum_multi(lists, FusionConfig::default())
                }
            }
            Self::Dbsf => crate::dbsf_multi(lists, FusionConfig::default()),
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
#[must_use]
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
    sort_scored_desc(output);
    if let Some(top_k) = config.top_k {
        output.truncate(top_k);
    }
}

/// RRF for 3+ result lists.
#[must_use]
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

/// Weighted RRF: per-retriever weights applied to rank-based scores.
///
/// Unlike standard RRF which treats all lists equally, weighted RRF allows
/// assigning different importance to different retrievers based on domain
/// knowledge or tuning.
///
/// Formula: `score(d) = Σ w_i / (k + rank_i(d))`
///
/// # Example
///
/// ```rust
/// use rank_fusion::{rrf_weighted, RrfConfig};
///
/// let bm25 = vec![("d1", 0.0), ("d2", 0.0)];   // scores ignored
/// let dense = vec![("d2", 0.0), ("d3", 0.0)];
///
/// // Trust dense retriever 2x more than BM25
/// let weights = [0.33, 0.67];
/// let fused = rrf_weighted(&[&bm25[..], &dense[..]], &weights, RrfConfig::default());
/// ```
///
/// # Errors
///
/// Returns [`FusionError::ZeroWeights`] if weights sum to zero.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_weighted<I, L>(lists: &[L], weights: &[f32], config: RrfConfig) -> Result<Vec<(I, f32)>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum.abs() < 1e-9 {
        return Err(FusionError::ZeroWeights);
    }

    let k = config.k as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();

    for (list, &weight) in lists.iter().zip(weights.iter()) {
        let normalized_weight = weight / weight_sum;
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            *scores.entry(id.clone()).or_default() += normalized_weight / (k + rank as f32);
        }
    }

    Ok(finalize(scores, config.top_k))
}

// ─────────────────────────────────────────────────────────────────────────────
// Score-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted score fusion with optional normalization.
///
/// Formula: `score(d) = w_a × norm(s_a) + w_b × norm(s_b)`
#[must_use]
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
#[must_use]
pub fn combsum<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combsum_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombSUM` with configuration.
#[must_use]
pub fn combsum_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combsum_multi(&[results_a, results_b], config)
}

/// `CombSUM` for 3+ result lists.
#[must_use]
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
#[must_use]
pub fn combmnz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmnz_with_config(results_a, results_b, FusionConfig::default())
}

/// `CombMNZ` with configuration.
#[must_use]
pub fn combmnz_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    combmnz_multi(&[results_a, results_b], config)
}

/// `CombMNZ` for 3+ result lists.
#[must_use]
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
    sort_scored_desc(&mut results);
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }
    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Rank-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Borda count — each position contributes `N - rank` points.
#[must_use]
pub fn borda<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    borda_with_config(results_a, results_b, FusionConfig::default())
}

/// Borda count with configuration.
#[must_use]
pub fn borda_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    borda_multi(&[results_a, results_b], config)
}

/// Borda count for 3+ result lists.
#[must_use]
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
// Distribution-Based Score Fusion (DBSF)
// ─────────────────────────────────────────────────────────────────────────────

/// Distribution-Based Score Fusion (DBSF).
///
/// Uses z-score normalization with mean ± 3σ clipping, then sums scores.
/// More robust than min-max normalization when score distributions differ.
///
/// # Algorithm
///
/// For each list:
/// 1. Compute mean (μ) and standard deviation (σ)
/// 2. Normalize: `n = (score - μ) / σ`, clipped to [-3, 3]
/// 3. Sum normalized scores across lists
///
/// # Example
///
/// ```rust
/// use rank_fusion::dbsf;
///
/// let bm25 = vec![("d1", 15.0), ("d2", 12.0), ("d3", 8.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.7), ("d4", 0.5)];
/// let fused = dbsf(&bm25, &dense);
/// ```
#[must_use]
pub fn dbsf<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    dbsf_with_config(results_a, results_b, FusionConfig::default())
}

/// DBSF with configuration.
#[must_use]
pub fn dbsf_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: FusionConfig,
) -> Vec<(I, f32)> {
    dbsf_multi(&[results_a, results_b], config)
}

/// DBSF for 3+ result lists.
#[must_use]
pub fn dbsf_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (id, s) in items {
            // Z-score normalize and clip to [-3, 3]
            let z = if std > 1e-9 {
                ((s - mean) / std).clamp(-3.0, 3.0)
            } else {
                0.0 // All scores equal
            };
            *scores.entry(id.clone()).or_default() += z;
        }
    }

    finalize(scores, config.top_k)
}

/// Compute mean and standard deviation for z-score normalization.
#[inline]
fn zscore_params<I>(results: &[(I, f32)]) -> (f32, f32) {
    if results.is_empty() {
        return (0.0, 1.0);
    }

    let n = results.len() as f32;
    let mean = results.iter().map(|(_, s)| s).sum::<f32>() / n;
    let variance = results.iter().map(|(_, s)| (s - mean).powi(2)).sum::<f32>() / n;
    let std = variance.sqrt();

    (mean, std)
}

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Sort scores descending and optionally truncate.
///
/// Uses `total_cmp` for deterministic NaN handling (NaN sorts after valid values).
#[inline]
fn finalize<I>(scores: HashMap<I, f32>, top_k: Option<usize>) -> Vec<(I, f32)> {
    let mut results: Vec<_> = scores.into_iter().collect();
    sort_scored_desc(&mut results);
    if let Some(k) = top_k {
        results.truncate(k);
    }
    results
}

/// Sort scored results in descending order.
///
/// Uses `f32::total_cmp` for deterministic ordering of NaN values.
#[inline]
fn sort_scored_desc<I>(results: &mut [(I, f32)]) {
    results.sort_by(|a, b| b.1.total_cmp(&a.1));
}

/// Returns (scale, offset) for min-max normalization: `(x - offset) * scale`.
#[inline]
/// Returns (norm_factor, offset) for min-max normalization.
///
/// Normalized score = (score - offset) * norm_factor
///
/// For single-element lists or lists where all scores are equal,
/// returns (0.0, 0.0) so each element contributes its raw score.
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
        // All scores equal: just pass through the score (norm=1, offset=0)
        (1.0, 0.0)
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
    fn rrf_weighted_applies_weights() {
        // d1 appears in list_a (rank 0), d2 appears in list_b (rank 0)
        let list_a = vec![("d1", 0.0)];
        let list_b = vec![("d2", 0.0)];

        // Weight list_b 3x more than list_a
        let weights = [0.25, 0.75];
        let f = rrf_weighted(&[&list_a[..], &list_b[..]], &weights, RrfConfig::new(60)).unwrap();

        // d2 should rank higher because its list has 3x the weight
        assert_eq!(f[0].0, "d2", "weighted RRF should favor higher-weight list");

        // Verify score formula: w / (k + rank)
        // d1: 0.25 / 60 = 0.00417
        // d2: 0.75 / 60 = 0.0125
        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let d2_score = f.iter().find(|(id, _)| *id == "d2").unwrap().1;
        assert!(
            d2_score > d1_score * 2.0,
            "d2 should score ~3x higher than d1"
        );
    }

    #[test]
    fn rrf_weighted_zero_weights_error() {
        let list_a = vec![("d1", 0.0)];
        let list_b = vec![("d2", 0.0)];
        let weights = [0.0, 0.0];

        let result = rrf_weighted(&[&list_a[..], &list_b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::ZeroWeights)));
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

    // ─────────────────────────────────────────────────────────────────────────
    // Edge Case Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn nan_scores_handled() {
        let a = vec![("d1", f32::NAN), ("d2", 0.5)];
        let b = vec![("d2", 0.9), ("d3", 0.1)];

        // Should not panic
        let _ = rrf(a.clone(), b.clone(), RrfConfig::default());
        let _ = combsum(&a, &b);
        let _ = combmnz(&a, &b);
        let _ = borda(&a, &b);
    }

    #[test]
    fn inf_scores_handled() {
        let a = vec![("d1", f32::INFINITY), ("d2", 0.5)];
        let b = vec![("d2", f32::NEG_INFINITY), ("d3", 0.1)];

        // Should not panic
        let _ = rrf(a.clone(), b.clone(), RrfConfig::default());
        let _ = combsum(&a, &b);
    }

    #[test]
    fn zero_scores() {
        let a = vec![("d1", 0.0), ("d2", 0.0)];
        let b = vec![("d2", 0.0), ("d3", 0.0)];

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn negative_scores() {
        let a = vec![("d1", -1.0), ("d2", -0.5)];
        let b = vec![("d2", -0.9), ("d3", -0.1)];

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 3);
        // Should normalize properly
    }

    #[test]
    fn large_k_value() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        // k = u32::MAX should not overflow
        let f = rrf(a, b, RrfConfig::new(u32::MAX));
        assert!(!f.is_empty());
    }

    #[test]
    fn single_item_lists() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d1", 1.0)];

        let f = rrf(a.clone(), b.clone(), RrfConfig::default());
        assert_eq!(f.len(), 1);

        let f = combsum(&a, &b);
        assert_eq!(f.len(), 1);

        let f = borda(&a, &b);
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn disjoint_lists() {
        let a = vec![("d1", 1.0), ("d2", 0.9)];
        let b = vec![("d3", 1.0), ("d4", 0.9)];

        let f = rrf(a.clone(), b.clone(), RrfConfig::default());
        assert_eq!(f.len(), 4);

        let f = combmnz(&a, &b);
        assert_eq!(f.len(), 4);
        // No overlap bonus
    }

    #[test]
    fn identical_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d1", "d2", "d3"]);

        let f = rrf(a.clone(), b.clone(), RrfConfig::default());
        // Order should be preserved
        assert_eq!(f[0].0, "d1");
        assert_eq!(f[1].0, "d2");
        assert_eq!(f[2].0, "d3");
    }

    #[test]
    fn reversed_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);

        let f = rrf(a, b, RrfConfig::default());
        // All items appear in both lists, so all have same total RRF score
        // d2 at rank 1 in both gets: 2 * 1/(60+1) = 2/61
        // d1 at rank 0,2 gets: 1/60 + 1/62
        // d3 at rank 2,0 gets: 1/62 + 1/60
        // d1 and d3 tie, d2 is slightly lower (rank 1+1 vs 0+2)
        // Just check we get all 3
        assert_eq!(f.len(), 3);
    }

    #[test]
    fn top_k_larger_than_result() {
        let a = ranked(&["d1"]);
        let b = ranked(&["d2"]);

        let f = rrf(a, b, RrfConfig::default().with_top_k(100));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn top_k_zero() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = rrf(a, b, RrfConfig::default().with_top_k(0));
        assert_eq!(f.len(), 0);
    }

    // ─────────────────────────────────────────────────────────────────────────
    // FusionMethod Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn fusion_method_rrf() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::rrf().fuse(&a, &b);
        assert_eq!(f[0].0, "d2"); // Appears in both
    }

    #[test]
    fn fusion_method_combsum() {
        // Use scores where d2 clearly wins after normalization
        // a: d1=1.0 (norm: 1.0), d2=0.5 (norm: 0.0)
        // b: d2=1.0 (norm: 1.0), d3=0.5 (norm: 0.0)
        // Final: d1=1.0, d2=1.0, d3=0.0 - still a tie!
        // Use 3 elements to break the tie:
        let a = vec![("d1", 1.0_f32), ("d2", 0.6), ("d4", 0.2)];
        let b = vec![("d2", 1.0_f32), ("d3", 0.5)];
        // a norms: d1=(1.0-0.2)/0.8=1.0, d2=(0.6-0.2)/0.8=0.5, d4=0.0
        // b norms: d2=(1.0-0.5)/0.5=1.0, d3=0.0
        // Final: d1=1.0, d2=0.5+1.0=1.5, d3=0.0, d4=0.0

        let f = FusionMethod::CombSum.fuse(&a, &b);
        // d2 appears in both lists with high scores, should win
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_combmnz() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::CombMnz.fuse(&a, &b);
        assert_eq!(f[0].0, "d2"); // Overlap bonus
    }

    #[test]
    fn fusion_method_borda() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::Borda.fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_weighted() {
        let a = vec![("d1", 1.0f32)];
        let b = vec![("d2", 1.0f32)];

        // Heavy weight on first list
        let f = FusionMethod::weighted(0.9, 0.1).fuse(&a, &b);
        assert_eq!(f[0].0, "d1");

        // Heavy weight on second list
        let f = FusionMethod::weighted(0.1, 0.9).fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_multi() {
        let lists: Vec<Vec<(&str, f32)>> = vec![
            ranked(&["d1", "d2"]),
            ranked(&["d2", "d3"]),
            ranked(&["d1", "d3"]),
        ];

        let f = FusionMethod::rrf().fuse_multi(&lists);
        assert_eq!(f.len(), 3);
        // d1 and d2 both appear in 2 lists, should be top 2
    }

    #[test]
    fn fusion_method_default_is_rrf() {
        let method = FusionMethod::default();
        assert!(matches!(method, FusionMethod::Rrf { k: 60 }));
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

        /// combmnz with overlap should score higher than without
        #[test]
        fn combmnz_overlap_bonus(id in 0u32..100, score in 0.1f32..1.0) {
            let a = vec![(id, score)];
            let b = vec![(id, score)];
            let c = vec![(id + 1, score)]; // different id, no overlap

            let overlapped = combmnz(&a, &b);
            let disjoint = combmnz(&a, &c);

            // Overlapped should have higher score (multiplied by 2)
            let overlap_score = overlapped.iter().find(|(i, _)| *i == id).map(|(_, s)| *s).unwrap_or(0.0);
            let disjoint_score = disjoint.iter().find(|(i, _)| *i == id).map(|(_, s)| *s).unwrap_or(0.0);
            prop_assert!(overlap_score >= disjoint_score);
        }

        /// weighted with extreme weights should favor one side
        #[test]
        fn weighted_extreme_weights(a_id in 0u32..50, b_id in 50u32..100) {
            let a = vec![(a_id, 1.0f32)];
            let b = vec![(b_id, 1.0f32)];

            let high_a = weighted(&a, &b, WeightedConfig::new(0.99, 0.01).with_normalize(false));
            let high_b = weighted(&a, &b, WeightedConfig::new(0.01, 0.99).with_normalize(false));

            // With single items and no normalization, the weighted score is just the weight
            // Both items get their respective weighted score, so higher weight wins
            let a_score_in_high_a = high_a.iter().find(|(id, _)| *id == a_id).map(|(_, s)| *s).unwrap_or(0.0);
            let b_score_in_high_a = high_a.iter().find(|(id, _)| *id == b_id).map(|(_, s)| *s).unwrap_or(0.0);
            prop_assert!(a_score_in_high_a > b_score_in_high_a);

            let a_score_in_high_b = high_b.iter().find(|(id, _)| *id == a_id).map(|(_, s)| *s).unwrap_or(0.0);
            let b_score_in_high_b = high_b.iter().find(|(id, _)| *id == b_id).map(|(_, s)| *s).unwrap_or(0.0);
            prop_assert!(b_score_in_high_b > a_score_in_high_b);
        }

        /// all algorithms should produce non-empty output for non-empty input
        #[test]
        fn nonempty_output(a in arb_results(5).prop_filter("need items", |v| !v.is_empty())) {
            let b: Vec<(u32, f32)> = vec![];

            prop_assert!(!rrf(a.clone(), b.clone(), RrfConfig::default()).is_empty());
            prop_assert!(!combsum(&a, &b).is_empty());
            prop_assert!(!combmnz(&a, &b).is_empty());
            prop_assert!(!borda(&a, &b).is_empty());
        }

        /// multi variants should match two-list for n=2 (same scores, may differ in order for ties)
        #[test]
        fn multi_matches_two_list(a in arb_results(10), b in arb_results(10)) {
            let two_list = rrf(a.clone(), b.clone(), RrfConfig::default());
            let multi = rrf_multi(&[a.clone(), b.clone()], RrfConfig::default());

            prop_assert_eq!(two_list.len(), multi.len());

            // Check same IDs with same scores (order may differ due to HashMap iteration)
            let two_map: HashMap<_, _> = two_list.into_iter().collect();
            let multi_map: HashMap<_, _> = multi.into_iter().collect();

            for (id, score) in &two_map {
                let multi_score = multi_map.get(id).expect("same ids");
                prop_assert!((score - multi_score).abs() < 1e-6, "score mismatch for {:?}", id);
            }
        }

        /// borda should give highest score to items in position 0
        #[test]
        fn borda_top_position_wins(n in 2usize..10) {
            let top_id = 999u32;
            let a: Vec<(u32, f32)> = std::iter::once((top_id, 1.0))
                .chain((0..n as u32 - 1).map(|i| (i, 0.9 - i as f32 * 0.1)))
                .collect();
            let b: Vec<(u32, f32)> = std::iter::once((top_id, 1.0))
                .chain((100..100 + n as u32 - 1).map(|i| (i, 0.9)))
                .collect();

            let f = borda(&a, &b);
            prop_assert_eq!(f[0].0, top_id);
        }

        // ─────────────────────────────────────────────────────────────────────────
        // NaN / Infinity / Edge Case Tests (learned from rank-refine)
        // ─────────────────────────────────────────────────────────────────────────

        /// NaN scores should not corrupt sort order
        #[test]
        fn nan_does_not_corrupt_sorting(a in arb_results(10)) {
            let mut with_nan = a.clone();
            if !with_nan.is_empty() {
                with_nan[0].1 = f32::NAN;
            }
            let b: Vec<(u32, f32)> = vec![];

            // Should not panic and should produce sorted output
            let result = combsum(&with_nan, &b);
            for window in result.windows(2) {
                // With total_cmp, NaN sorts consistently
                let cmp = window[0].1.total_cmp(&window[1].1);
                prop_assert!(cmp != std::cmp::Ordering::Less,
                    "Not sorted: {:?} < {:?}", window[0].1, window[1].1);
            }
        }

        /// Infinity scores should be handled gracefully (no panics)
        #[test]
        fn infinity_handled_gracefully(id in 0u32..50) {
            let a = vec![(id, f32::INFINITY)];
            let b = vec![(id + 100, f32::NEG_INFINITY)]; // Different ID to avoid collision

            // Should not panic
            let result = combsum(&a, &b);
            prop_assert_eq!(result.len(), 2);
            // Just verify we got results, don't assert order (normalization changes things)
        }

        /// Output always sorted descending (invariant)
        #[test]
        fn output_always_sorted(a in arb_results(20), b in arb_results(20)) {
            for result in [
                rrf(a.clone(), b.clone(), RrfConfig::default()),
                combsum(&a, &b),
                combmnz(&a, &b),
                borda(&a, &b),
            ] {
                for window in result.windows(2) {
                    prop_assert!(
                        window[0].1.total_cmp(&window[1].1) != std::cmp::Ordering::Less,
                        "Not sorted: {} < {}", window[0].1, window[1].1
                    );
                }
            }
        }

        /// Unique IDs in output (no duplicates)
        #[test]
        fn unique_ids_in_output(a in arb_results(20), b in arb_results(20)) {
            let result = rrf(a, b, RrfConfig::default());
            let mut seen = std::collections::HashSet::new();
            for (id, _) in &result {
                prop_assert!(seen.insert(id), "Duplicate ID in output: {:?}", id);
            }
        }

        /// CombSUM scores are non-negative after normalization
        #[test]
        fn combsum_scores_nonnegative(a in arb_results(10), b in arb_results(10)) {
            let result = combsum(&a, &b);
            for (_, score) in &result {
                if !score.is_nan() {
                    prop_assert!(*score >= -0.01, "Score {} is negative", score);
                }
            }
        }

        /// Equal weights produce symmetric treatment
        #[test]
        fn equal_weights_symmetric(a in arb_results(10), b in arb_results(10)) {
            let ab = weighted(&a, &b, WeightedConfig::default());
            let ba = weighted(&b, &a, WeightedConfig::default());

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                if let Some(score_ba) = ba_map.get(id) {
                    prop_assert!((score_ab - score_ba).abs() < 1e-5,
                        "Symmetric treatment violated for {:?}: {} != {}", id, score_ab, score_ba);
                }
            }
        }

        /// RRF scores bounded by 2/k for items in both lists
        #[test]
        fn rrf_score_bounded(k in 1u32..1000) {
            let a = vec![(1u32, 1.0)];
            let b = vec![(1u32, 1.0)];

            let result = rrf(a, b, RrfConfig::new(k));
            let max_possible = 2.0 / k as f32; // rank 0 in both lists
            prop_assert!(result[0].1 <= max_possible + 1e-6);
        }

        /// Empty list handling: combining with empty should equal single list
        #[test]
        fn empty_list_preserves_ids(n in 1usize..10) {
            // Create list with unique IDs
            let a: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, 1.0 - i as f32 * 0.1)).collect();
            let empty: Vec<(u32, f32)> = vec![];

            let rrf_result = rrf(a.clone(), empty, RrfConfig::default());

            // Should have same number of unique IDs
            prop_assert_eq!(rrf_result.len(), n);

            // All original IDs should be present
            for (id, _) in &a {
                prop_assert!(rrf_result.iter().any(|(rid, _)| rid == id), "Missing ID {:?}", id);
            }
        }
    }
}
