//! Rank fusion for hybrid search.
//!
//! Combine results from multiple retrievers (BM25, dense, sparse) into a single ranking.
//!
//! ```rust
//! use rank_fusion::rrf;
//!
//! let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
//! let dense = vec![("d2", 0.9), ("d3", 0.8)];
//! let fused = rrf(&bm25, &dense);
//! // d2 ranks highest (appears in both lists)
//! ```
//!
//! # Algorithms
//!
//! | Function | Uses Scores | Best For |
//! |----------|-------------|----------|
//! | [`rrf`] | No | Incompatible score scales |
//! | [`isr`] | No | When lower ranks matter more |
//! | [`combsum`] | Yes | Similar scales, trust scores |
//! | [`combmnz`] | Yes | Reward overlap between lists |
//! | [`borda`] | No | Simple voting |
//! | [`weighted`] | Yes | Custom retriever weights |
//! | [`dbsf`] | Yes | Different score distributions |
//!
//! All have `*_multi` variants for 3+ lists.
//!
//! # Performance Notes
//!
//! `OpenSearch` benchmarks (BEIR) show RRF is ~3-4% lower NDCG than score-based
//! fusion (`CombSUM`), but ~1-2% faster. RRF excels when score scales are
//! incompatible or unknown. See [OpenSearch RRF blog](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/).

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

/// Threshold for treating weight sum as effectively zero.
///
/// Used in weighted fusion to detect invalid configurations where all weights
/// are zero or near-zero, which would cause division by zero.
const WEIGHT_EPSILON: f32 = 1e-9;

/// Threshold for treating score range as effectively zero (all scores equal).
///
/// Used in min-max normalization to detect degenerate cases where all scores
/// are identical, avoiding division by zero.
const SCORE_RANGE_EPSILON: f32 = 1e-9;

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
    ///
    /// **Must be >= 1** to avoid division by zero in the RRF formula.
    /// Values < 1 will cause panics during fusion.
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
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` (would cause division by zero in RRF formula).
    ///
    /// # Example
    ///
    /// ```rust
    /// use rank_fusion::RrfConfig;
    ///
    /// let config = RrfConfig::new(60);
    /// ```
    #[must_use]
    pub fn new(k: u32) -> Self {
        assert!(
            k >= 1,
            "k must be >= 1 to avoid division by zero in RRF formula"
        );
        Self { k, top_k: None }
    }

    /// Set the k parameter (smoothing constant).
    ///
    /// - `k=60` — Standard RRF, works well for most cases
    /// - `k=1` — Top positions dominate heavily
    /// - `k=100+` — More uniform contribution across ranks
    ///
    /// # Panics
    ///
    /// Panics if `k == 0` (would cause division by zero in RRF formula).
    #[must_use]
    pub fn with_k(mut self, k: u32) -> Self {
        assert!(
            k >= 1,
            "k must be >= 1 to avoid division by zero in RRF formula"
        );
        self.k = k;
        self
    }

    /// Limit output to `top_k` results.
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

    /// Limit output to `top_k` results.
    #[must_use]
    pub const fn with_top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }
}

/// Configuration for rank-based fusion (Borda, `CombSUM`, `CombMNZ`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct FusionConfig {
    /// Maximum results to return (None = all).
    pub top_k: Option<usize>,
}

impl FusionConfig {
    /// Limit output to `top_k` results.
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
    pub use crate::{
        borda, combanz, combmax, combmed, combmnz, combsum, dbsf, isr, isr_with_config, rrf,
        rrf_with_config, weighted,
    };
    pub use crate::{
        FusionConfig, FusionError, FusionMethod, Normalization, Result, RrfConfig, WeightedConfig,
    };
}

/// Explainability module for debugging and analysis.
///
/// Provides variants of fusion functions that return full provenance information,
/// showing which retrievers contributed each document and how scores were computed.
pub mod explain {
    pub use crate::{
        analyze_consensus, attribute_top_k, combmnz_explain, combsum_explain, dbsf_explain,
        rrf_explain, ConsensusReport, Explanation, FusedResult, RetrieverId, RetrieverStats,
        SourceContribution,
    };
}

/// Strategy module for runtime fusion method selection.
///
/// Enables dynamic selection of fusion methods without trait objects.
pub mod strategy {
    pub use crate::FusionStrategy;
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
    /// Inverse Square Root rank fusion (gentler decay than RRF).
    Isr {
        /// Smoothing constant (default: 1).
        k: u32,
    },
    /// `CombSUM` — sum of normalized scores.
    CombSum,
    /// `CombMNZ` — sum × overlap count.
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

    /// Create ISR method with default k=1.
    #[must_use]
    pub const fn isr() -> Self {
        Self::Isr { k: 1 }
    }

    /// Create ISR method with custom k.
    #[must_use]
    pub const fn isr_with_k(k: u32) -> Self {
        Self::Isr { k }
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
            Self::Rrf { k } => {
                // Validate k at use time to avoid panics from invalid FusionMethod construction
                if *k == 0 {
                    return Vec::new();
                }
                crate::rrf_multi(&[a, b], RrfConfig::new(*k))
            }
            Self::Isr { k } => {
                if *k == 0 {
                    return Vec::new();
                }
                crate::isr_multi(&[a, b], RrfConfig::new(*k))
            }
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
            Self::Isr { k } => crate::isr_multi(lists, RrfConfig::new(*k)),
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

/// Reciprocal Rank Fusion of two result lists with default config (k=60).
///
/// Formula: `score(d) = Σ 1/(k + rank)` where rank is 0-indexed.
///
/// **Why RRF?** Different retrievers use incompatible score scales (BM25: 0-100,
/// dense: 0-1). RRF solves this by ignoring scores entirely and using only rank
/// positions. The reciprocal formula ensures:
/// - Top positions dominate (rank 0 gets 1/60 = 0.017, rank 5 gets 1/65 = 0.015)
/// - Multiple list agreement is rewarded (documents appearing in both lists score higher)
/// - No normalization needed (works with any score distribution)
///
/// **When to use**: Hybrid search with incompatible score scales, zero-configuration needs.
/// **When NOT to use**: When score scales are compatible, CombSUM achieves ~3-4% better NDCG.
///
/// Use [`rrf_with_config`] to customize the k parameter (lower k = more top-heavy).
///
/// # Duplicate Document IDs
///
/// If a document ID appears multiple times in the same list, **all occurrences contribute**
/// to the RRF score based on their respective ranks. For example, if "d1" appears at
/// rank 0 and rank 5 in list A, its contribution from list A is `1/(k+0) + 1/(k+5)`.
/// This differs from some implementations that take only the first occurrence.
///
/// # Complexity
///
/// O(n log n) where n = |a| + |b| (dominated by final sort).
///
/// # Example
///
/// ```rust
/// use rank_fusion::rrf;
///
/// let sparse = vec![("d1", 0.9), ("d2", 0.5)];
/// let dense = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = rrf(&sparse, &dense);
/// assert_eq!(fused[0].0, "d2"); // appears in both lists (consensus)
/// ```
#[must_use]
pub fn rrf<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    rrf_with_config(results_a, results_b, RrfConfig::default())
}

/// RRF with custom configuration.
///
/// Use this when you need to tune the k parameter:
/// - **k=20-40**: Top positions dominate more. Use when top retrievers are highly reliable.
/// - **k=60**: Default (empirically chosen by Cormack et al., 2009). Balanced for most scenarios.
/// - **k=100+**: More uniform contribution. Use when lower-ranked items are still valuable.
///
/// **Sensitivity**: k=10 gives 1.5x ratio (rank 0 vs rank 5), k=60 gives 1.1x, k=100 gives 1.05x.
///
/// # Example
///
/// ```rust
/// use rank_fusion::{rrf_with_config, RrfConfig};
///
/// let a = vec![("d1", 0.9), ("d2", 0.5)];
/// let b = vec![("d2", 0.8), ("d3", 0.3)];
///
/// // k=20: emphasize top positions (strong consensus required)
/// let fused = rrf_with_config(&a, &b, RrfConfig::new(20));
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
) -> Vec<(I, f32)> {
    let k = config.k as f32;
    // Pre-allocate capacity to avoid reallocations during insertion
    let estimated_size = results_a.len() + results_b.len();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
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

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32);
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }

    output.extend(scores);
    sort_scored_desc(output);
    if let Some(top_k) = config.top_k {
        output.truncate(top_k);
    }
}

/// RRF for 3+ result lists.
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. If some lists are empty,
/// they contribute zero scores (documents not appearing in those lists
/// receive no contribution from them).
///
/// # Complexity
///
/// O(L×N + U×log U) where L = number of lists, N = average list size,
/// U = number of unique document IDs across all lists.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_multi<I, L>(lists: &[L], config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let k = config.k as f32;
    // Estimate capacity: sum of all list sizes (may overestimate due to duplicates)
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for list in lists {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32);
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
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
/// - Returns [`FusionError::ZeroWeights`] if weights sum to zero.
/// - Returns [`FusionError::InvalidConfig`] if `lists.len() != weights.len()`.
#[allow(clippy::cast_precision_loss)]
pub fn rrf_weighted<I, L>(lists: &[L], weights: &[f32], config: RrfConfig) -> Result<Vec<(I, f32)>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.len() != weights.len() {
        return Err(FusionError::InvalidConfig(format!(
            "lists.len() ({}) != weights.len() ({}). Each list must have a corresponding weight.",
            lists.len(),
            weights.len()
        )));
    }
    let weight_sum: f32 = weights.iter().sum();
    if weight_sum.abs() < WEIGHT_EPSILON {
        return Err(FusionError::ZeroWeights);
    }

    let k = config.k as f32;
    // Pre-allocate capacity
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, &weight) in lists.iter().zip(weights.iter()) {
        let normalized_weight = weight / weight_sum;
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = normalized_weight / (k + rank as f32);
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    Ok(finalize(scores, config.top_k))
}

// ─────────────────────────────────────────────────────────────────────────────
// ISR (Inverse Square Root Rank)
// ─────────────────────────────────────────────────────────────────────────────

/// Inverse Square Root rank fusion with default config (k=1).
///
/// Formula: `score(d) = Σ 1/sqrt(k + rank)` where rank is 0-indexed.
///
/// ISR has a gentler decay than RRF — lower ranks contribute more relative
/// to top positions. Useful when you believe relevant documents may appear
/// deeper in the lists.
///
/// Use [`isr_with_config`] to customize the k parameter.
///
/// # Complexity
///
/// O(n log n) where n = |a| + |b| (dominated by final sort).
///
/// # Example
///
/// ```rust
/// use rank_fusion::isr;
///
/// let sparse = vec![("d1", 0.9), ("d2", 0.5)];
/// let dense = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = isr(&sparse, &dense);
/// assert_eq!(fused[0].0, "d2"); // appears in both lists
/// ```
#[must_use]
pub fn isr<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    isr_with_config(results_a, results_b, RrfConfig::new(1))
}

/// ISR with custom configuration.
///
/// The k parameter controls decay steepness:
/// - Lower k (e.g., 1): Top positions dominate more
/// - Higher k (e.g., 10): More uniform contribution across positions
///
/// # Example
///
/// ```rust
/// use rank_fusion::{isr_with_config, RrfConfig};
///
/// let a = vec![("d1", 0.9), ("d2", 0.5)];
/// let b = vec![("d2", 0.8), ("d3", 0.3)];
///
/// let fused = isr_with_config(&a, &b, RrfConfig::new(1));
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn isr_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: RrfConfig,
) -> Vec<(I, f32)> {
    let k = config.k as f32;
    let estimated_size = results_a.len() + results_b.len();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
    for (rank, (id, _)) in results_a.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32).sqrt();
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }
    for (rank, (id, _)) in results_b.iter().enumerate() {
        let contribution = 1.0 / (k + rank as f32).sqrt();
        if let Some(score) = scores.get_mut(id) {
            *score += contribution;
        } else {
            scores.insert(id.clone(), contribution);
        }
    }

    finalize(scores, config.top_k)
}

/// ISR for 3+ result lists.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn isr_multi<I, L>(lists: &[L], config: RrfConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let k = config.k as f32;
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32).sqrt();
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Score-based Fusion
// ─────────────────────────────────────────────────────────────────────────────

/// Weighted score fusion with configurable retriever trust.
///
/// Formula: `score(d) = w_a × norm(s_a) + w_b × norm(s_b)`
///
/// Use when you know one retriever is more reliable for your domain.
/// Weights are normalized to sum to 1.
///
/// # Complexity
///
/// O(n log n) where n = total items across all lists.
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
    if total_weight.abs() < WEIGHT_EPSILON {
        return Err(FusionError::ZeroWeights);
    }

    let estimated_size: usize = lists.iter().map(|(l, _)| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            let contribution = w * (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
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
    if total_weight.abs() < WEIGHT_EPSILON {
        return Vec::new();
    }

    let estimated_size: usize = lists.iter().map(|(l, _)| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, weight) in lists {
        let items = list.as_ref();
        let w = weight / total_weight;
        let (norm, off) = if normalize {
            min_max_params(items)
        } else {
            (1.0, 0.0)
        };
        for (id, s) in items {
            let contribution = w * (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, top_k)
}

/// Sum of min-max normalized scores.
///
/// Formula: `score(d) = Σ (s - min) / (max - min)`
///
/// Use when scores are on similar scales (e.g., all cosine similarities).
///
/// # Complexity
///
/// O(n log n) where n = total items across all lists.
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
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no contribution from them).
#[must_use]
pub fn combsum_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            let contribution = (s - off) * norm;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
        }
    }

    finalize(scores, config.top_k)
}

/// Normalized sum × overlap count (rewards agreement).
///
/// Formula: `score(d) = CombSUM(d) × |{lists containing d}|`
///
/// Documents appearing in more lists get a multiplier bonus.
/// Use when overlap signals higher relevance.
///
/// # Complexity
///
/// O(n log n) where n = total items across all lists.
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
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores and don't affect overlap counts.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combmnz_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, (f32, u32)> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (id, s) in items {
            // Use get_mut + insert pattern to avoid cloning IDs when entry already exists
            let contribution = (s - off) * norm;
            if let Some(entry) = scores.get_mut(id) {
                entry.0 += contribution;
                entry.1 += 1;
            } else {
                scores.insert(id.clone(), (contribution, 1));
            }
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

/// Borda count voting — position-based scoring.
///
/// Formula: `score(d) = Σ (N - rank)` where N = list length, rank is 0-indexed.
///
/// Ignores original scores; only considers position. Simple and robust
/// when you don't trust score magnitudes.
///
/// # Complexity
///
/// O(n log n) where n = total items across all lists.
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
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no Borda points from them).
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn borda_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let n = items.len() as f32;
        for (rank, (id, _)) in items.iter().enumerate() {
            let contribution = n - rank as f32;
            if let Some(score) = scores.get_mut(id) {
                *score += contribution;
            } else {
                scores.insert(id.clone(), contribution);
            }
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
///
/// # Empty Lists
///
/// If `lists` is empty, returns an empty result. Empty lists within the slice
/// contribute zero scores (documents not appearing in those lists receive
/// no z-score contribution from them).
///
/// # Degenerate Cases
///
/// If all scores in a list are equal (zero variance), that list contributes
/// z-score=0.0 for all documents, which is mathematically correct but
/// effectively ignores that list's contribution.
#[must_use]
pub fn dbsf_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let estimated_size: usize = lists.iter().map(|l| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for list in lists {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (id, s) in items {
            // Z-score normalize and clip to [-3, 3]
            let z = if std > SCORE_RANGE_EPSILON {
                ((s - mean) / std).clamp(-3.0, 3.0)
            } else {
                0.0 // All scores equal
            };
            if let Some(score) = scores.get_mut(id) {
                *score += z;
            } else {
                scores.insert(id.clone(), z);
            }
        }
    }

    finalize(scores, config.top_k)
}

/// Compute mean and standard deviation for z-score normalization.
#[inline(always)]
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
    let capacity = top_k.map(|k| k.min(scores.len())).unwrap_or(scores.len());
    let mut results = Vec::with_capacity(capacity);
    results.extend(scores);
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

/// Score normalization methods.
///
/// Different retrievers produce scores on different scales. Normalization
/// puts them on a common scale before combining.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Normalization {
    /// Min-max normalization: `(score - min) / (max - min)` → [0, 1]
    ///
    /// Best when score distributions are similar. Sensitive to outliers.
    #[default]
    MinMax,
    /// Z-score normalization: `(score - mean) / std`, clipped to [-3, 3]
    ///
    /// More robust to outliers. Better when distributions differ.
    ZScore,
    /// Sum normalization: `score / sum(scores)`
    ///
    /// Preserves relative magnitudes. Useful when scores represent probabilities.
    Sum,
    /// Rank-based: convert scores to ranks, then normalize
    ///
    /// Ignores score magnitudes entirely. Most robust but loses information.
    Rank,
    /// No normalization: use raw scores
    ///
    /// Only use when all retrievers use the same scale.
    None,
}

/// Normalize a list of scores using the specified method.
///
/// Returns a vector of (id, normalized_score) pairs.
pub fn normalize_scores<I: Clone>(results: &[(I, f32)], method: Normalization) -> Vec<(I, f32)> {
    if results.is_empty() {
        return Vec::new();
    }

    match method {
        Normalization::MinMax => {
            let (norm, off) = min_max_params(results);
            results
                .iter()
                .map(|(id, s)| (id.clone(), (s - off) * norm))
                .collect()
        }
        Normalization::ZScore => {
            let (mean, std) = zscore_params(results);
            results
                .iter()
                .map(|(id, s)| {
                    let z = if std > SCORE_RANGE_EPSILON {
                        ((s - mean) / std).clamp(-3.0, 3.0)
                    } else {
                        0.0
                    };
                    (id.clone(), z)
                })
                .collect()
        }
        Normalization::Sum => {
            let sum: f32 = results.iter().map(|(_, s)| s).sum();
            if sum.abs() < SCORE_RANGE_EPSILON {
                return results.to_vec();
            }
            results
                .iter()
                .map(|(id, s)| (id.clone(), s / sum))
                .collect()
        }
        Normalization::Rank => {
            // Convert to ranks (0-indexed), then normalize by list length
            let n = results.len() as f32;
            results
                .iter()
                .enumerate()
                .map(|(rank, (id, _))| (id.clone(), 1.0 - (rank as f32 / n)))
                .collect()
        }
        Normalization::None => results.to_vec(),
    }
}

/// Returns `(norm_factor, offset)` for min-max normalization.
///
/// Normalized score = `(score - offset) * norm_factor`
///
/// For single-element lists or lists where all scores are equal,
/// returns `(0.0, 0.0)` so each element contributes its raw score.
#[inline(always)]
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
    if range < SCORE_RANGE_EPSILON {
        // All scores equal: just pass through the score (norm=1, offset=0)
        (1.0, 0.0)
    } else {
        (1.0 / range, min)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explainability
// ─────────────────────────────────────────────────────────────────────────────

/// A fused result with full provenance information for debugging and analysis.
///
/// Unlike the simple `Vec<(DocId, f32)>` returned by standard fusion functions,
/// `FusedResult` preserves which retrievers contributed each document, their
/// original ranks and scores, and how much each source contributed to the final score.
///
/// # Example
///
/// ```rust
/// use rank_fusion::explain::{rrf_explain, RetrieverId};
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let retrievers = vec![
///     RetrieverId::new("bm25"),
///     RetrieverId::new("dense"),
/// ];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &retrievers,
///     rank_fusion::RrfConfig::default(),
/// );
///
/// // d2 appears in both lists, so it has 2 source contributions
/// let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
/// assert_eq!(d2.explanation.sources.len(), 2);
/// assert_eq!(d2.explanation.consensus_score, 1.0); // 2/2 lists
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct FusedResult<K> {
    /// Document identifier.
    pub id: K,
    /// Final fused score.
    pub score: f32,
    /// Final rank position (0-indexed, highest score = rank 0).
    pub rank: usize,
    /// Explanation of how this score was computed.
    pub explanation: Explanation,
}

/// Explanation of how a fused score was computed.
#[derive(Debug, Clone, PartialEq)]
pub struct Explanation {
    /// Contributions from each retriever that contained this document.
    pub sources: Vec<SourceContribution>,
    /// Fusion method used (e.g., "rrf", "combsum").
    pub method: &'static str,
    /// Consensus score: fraction of retrievers that contained this document (0.0-1.0).
    ///
    /// - 1.0 = document appeared in all retrievers (strong consensus)
    /// - 0.5 = document appeared in half of retrievers
    /// - < 0.3 = document appeared in few retrievers (outlier)
    pub consensus_score: f32,
}

/// Contribution from a single retriever to a document's final score.
#[derive(Debug, Clone, PartialEq)]
pub struct SourceContribution {
    /// Identifier for this retriever (e.g., "bm25", "dense_vector").
    pub retriever_id: String,
    /// Original rank in this retriever's list (0-indexed, None if not present).
    pub original_rank: Option<usize>,
    /// Original score from this retriever (None for rank-based methods or if not present).
    pub original_score: Option<f32>,
    /// Normalized score (for score-based methods, None for rank-based).
    pub normalized_score: Option<f32>,
    /// How much this source contributed to the final fused score.
    ///
    /// For RRF: `1/(k + rank)` or `weight / (k + rank)` for weighted.
    /// For CombSUM: normalized score.
    /// For CombMNZ: normalized score × overlap count.
    pub contribution: f32,
}

/// Retriever identifier for explainability.
///
/// Used to label which retriever each list comes from when calling explain variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RetrieverId {
    id: String,
}

impl RetrieverId {
    /// Create a new retriever identifier.
    pub fn new<S: Into<String>>(id: S) -> Self {
        Self { id: id.into() }
    }

    /// Get the identifier string.
    pub fn as_str(&self) -> &str {
        &self.id
    }
}

impl From<&str> for RetrieverId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for RetrieverId {
    fn from(id: String) -> Self {
        Self::new(id)
    }
}

/// RRF with explainability: returns full provenance for each result.
///
/// This variant preserves which retrievers contributed each document, their
/// original ranks, and how much each source contributed to the final RRF score.
///
/// # Arguments
///
/// * `lists` - Ranked lists from each retriever
/// * `retriever_ids` - Identifiers for each retriever (must match `lists.len()`)
/// * `config` - RRF configuration
///
/// # Returns
///
/// Results sorted by fused score (descending), with full explanation metadata.
///
/// # Example
///
/// ```rust
/// use rank_fusion::explain::{rrf_explain, RetrieverId};
/// use rank_fusion::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let retrievers = vec![
///     RetrieverId::new("bm25"),
///     RetrieverId::new("dense"),
/// ];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &retrievers,
///     RrfConfig::default(),
/// );
///
/// // d2 appears in both lists at rank 1 and 0 respectively
/// let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
/// assert_eq!(d2.explanation.sources.len(), 2);
/// assert_eq!(d2.explanation.consensus_score, 1.0); // in both lists
///
/// // Check contributions
/// let bm25_contrib = d2.explanation.sources.iter()
///     .find(|s| s.retriever_id == "bm25")
///     .unwrap();
/// assert_eq!(bm25_contrib.original_rank, Some(1));
/// assert!(bm25_contrib.contribution > 0.0);
/// ```
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rrf_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: RrfConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let k = config.k as f32;
    let num_retrievers = lists.len() as f32;

    // Track scores and provenance
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        for (rank, (id, original_score)) in list.as_ref().iter().enumerate() {
            let contribution = 1.0 / (k + rank as f32);

            // Update score
            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            // Track provenance
            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: None, // RRF doesn't normalize
                    contribution,
                });
        }
    }

    // Build results with explanations
    let mut results: Vec<FusedResult<I>> = scores
        .into_iter()
        .map(|(id, score)| {
            let sources = provenance.remove(&id).unwrap_or_default();
            let consensus_score = sources.len() as f32 / num_retrievers;

            FusedResult {
                id,
                score,
                rank: 0, // Will be set after sorting
                explanation: Explanation {
                    sources,
                    method: "rrf",
                    consensus_score,
                },
            }
        })
        .collect();

    // Sort by score descending
    results.sort_by(|a, b| b.score.total_cmp(&a.score));

    // Set ranks
    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    // Apply top_k
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }

    results
}

/// Analyze consensus patterns across retrievers.
///
/// Returns statistics about how retrievers agree or disagree on document relevance.
///
/// # Example
///
/// ```rust
/// use rank_fusion::explain::{rrf_explain, analyze_consensus, RetrieverId};
/// use rank_fusion::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
///     RrfConfig::default(),
/// );
///
/// let consensus = analyze_consensus(&explained);
/// // consensus.high_consensus contains documents in all retrievers
/// // consensus.single_source contains documents only in one retriever
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct ConsensusReport<K> {
    /// Documents that appeared in all retrievers (consensus_score == 1.0).
    pub high_consensus: Vec<K>,
    /// Documents that appeared in only one retriever (consensus_score < 0.5).
    pub single_source: Vec<K>,
    /// Documents with large rank disagreements across retrievers.
    ///
    /// A document might appear at rank 0 in one retriever but rank 50 in another,
    /// indicating retriever disagreement.
    pub rank_disagreement: Vec<(K, Vec<(String, usize)>)>,
}

pub fn analyze_consensus<K: Clone + Eq + Hash>(results: &[FusedResult<K>]) -> ConsensusReport<K> {
    let mut high_consensus = Vec::new();
    let mut single_source = Vec::new();
    let mut rank_disagreement = Vec::new();

    for result in results {
        // High consensus: in all retrievers
        if result.explanation.consensus_score >= 1.0 - 1e-6 {
            high_consensus.push(result.id.clone());
        }

        // Single source: in only one retriever
        if result.explanation.sources.len() == 1 {
            single_source.push(result.id.clone());
        }

        // Rank disagreement: large spread in ranks
        if result.explanation.sources.len() > 1 {
            let ranks: Vec<usize> = result
                .explanation
                .sources
                .iter()
                .filter_map(|s| s.original_rank)
                .collect();
            if let (Some(&min_rank), Some(&max_rank)) = (ranks.iter().min(), ranks.iter().max()) {
                if max_rank - min_rank > 10 {
                    // Large disagreement threshold
                    let rank_info: Vec<(String, usize)> = result
                        .explanation
                        .sources
                        .iter()
                        .filter_map(|s| s.original_rank.map(|r| (s.retriever_id.clone(), r)))
                        .collect();
                    rank_disagreement.push((result.id.clone(), rank_info));
                }
            }
        }
    }

    ConsensusReport {
        high_consensus,
        single_source,
        rank_disagreement,
    }
}

/// Attribution statistics for each retriever.
///
/// Shows how much each retriever contributed to the top-k results.
#[derive(Debug, Clone, PartialEq)]
pub struct RetrieverStats {
    /// Number of top-k documents this retriever contributed.
    pub top_k_count: usize,
    /// Average contribution strength for documents in top-k.
    pub avg_contribution: f32,
    /// Documents that only this retriever found (unique to this retriever).
    pub unique_docs: usize,
}

/// Attribute top-k results to retrievers.
///
/// Returns statistics showing which retrievers contributed most to the top-k results.
///
/// # Example
///
/// ```rust
/// use rank_fusion::explain::{rrf_explain, attribute_top_k, RetrieverId};
/// use rank_fusion::RrfConfig;
///
/// let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
/// let dense = vec![("d2", 0.9), ("d3", 0.8)];
///
/// let explained = rrf_explain(
///     &[&bm25[..], &dense[..]],
///     &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
///     RrfConfig::default(),
/// );
///
/// let attribution = attribute_top_k(&explained, 5);
/// // attribution["bm25"].top_k_count shows how many top-5 docs came from BM25
/// ```
pub fn attribute_top_k<K: Clone + Eq + Hash>(
    results: &[FusedResult<K>],
    k: usize,
) -> std::collections::HashMap<String, RetrieverStats> {
    let top_k = results.iter().take(k);
    let mut stats: std::collections::HashMap<String, RetrieverStats> =
        std::collections::HashMap::new();

    // Track which documents each retriever found
    let mut retriever_docs: std::collections::HashMap<String, std::collections::HashSet<K>> =
        std::collections::HashMap::new();

    for result in top_k {
        for source in &result.explanation.sources {
            let entry =
                stats
                    .entry(source.retriever_id.clone())
                    .or_insert_with(|| RetrieverStats {
                        top_k_count: 0,
                        avg_contribution: 0.0,
                        unique_docs: 0,
                    });

            entry.top_k_count += 1;
            entry.avg_contribution += source.contribution;

            retriever_docs
                .entry(source.retriever_id.clone())
                .or_default()
                .insert(result.id.clone());
        }
    }

    // Calculate averages and unique counts
    for (retriever_id, stat) in &mut stats {
        if stat.top_k_count > 0 {
            stat.avg_contribution /= stat.top_k_count as f32;
        }

        // Count unique documents (only in this retriever)
        let this_retriever_docs = retriever_docs
            .get(retriever_id)
            .cloned()
            .unwrap_or_default();
        let other_retriever_docs: std::collections::HashSet<K> = retriever_docs
            .iter()
            .filter(|(id, _)| *id != retriever_id)
            .flat_map(|(_, docs)| docs.iter().cloned())
            .collect();

        stat.unique_docs = this_retriever_docs
            .difference(&other_retriever_docs)
            .count();
    }

    stats
}

/// CombSUM with explainability.
#[must_use]
pub fn combsum_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (rank, (id, original_score)) in items.iter().enumerate() {
            let normalized_score = (original_score - off) * norm;
            let contribution = normalized_score;

            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(normalized_score),
                    contribution,
                });
        }
    }

    build_explained_results(scores, provenance, num_retrievers, "combsum", config.top_k)
}

/// CombMNZ with explainability.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combmnz_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, (f32, u32)> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (norm, off) = min_max_params(items);
        for (rank, (id, original_score)) in items.iter().enumerate() {
            let normalized_score = (original_score - off) * norm;
            let contribution = normalized_score;

            let entry = scores.entry(id.clone()).or_insert((0.0, 0));
            entry.0 += contribution;
            entry.1 += 1;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(normalized_score),
                    contribution,
                });
        }
    }

    // Apply CombMNZ multiplier (overlap count)
    let mut final_scores: HashMap<I, f32> = HashMap::new();
    let mut final_provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (id, (sum, overlap_count)) in scores {
        let final_score = sum * overlap_count as f32;
        final_scores.insert(id.clone(), final_score);

        // Update contributions to reflect multiplier
        if let Some(mut sources) = provenance.remove(&id) {
            for source in &mut sources {
                source.contribution *= overlap_count as f32;
            }
            final_provenance.insert(id, sources);
        }
    }

    build_explained_results(
        final_scores,
        final_provenance,
        num_retrievers,
        "combmnz",
        config.top_k,
    )
}

/// DBSF with explainability.
#[must_use]
pub fn dbsf_explain<I, L>(
    lists: &[L],
    retriever_ids: &[RetrieverId],
    config: FusionConfig,
) -> Vec<FusedResult<I>>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() || lists.len() != retriever_ids.len() {
        return Vec::new();
    }

    let num_retrievers = lists.len() as f32;
    let mut scores: HashMap<I, f32> = HashMap::new();
    let mut provenance: HashMap<I, Vec<SourceContribution>> = HashMap::new();

    for (list, retriever_id) in lists.iter().zip(retriever_ids.iter()) {
        let items = list.as_ref();
        let (mean, std) = zscore_params(items);

        for (rank, (id, original_score)) in items.iter().enumerate() {
            let z = if std > SCORE_RANGE_EPSILON {
                ((original_score - mean) / std).clamp(-3.0, 3.0)
            } else {
                0.0
            };
            let contribution = z;

            *scores.entry(id.clone()).or_insert(0.0) += contribution;

            provenance
                .entry(id.clone())
                .or_default()
                .push(SourceContribution {
                    retriever_id: retriever_id.id.clone(),
                    original_rank: Some(rank),
                    original_score: Some(*original_score),
                    normalized_score: Some(z),
                    contribution,
                });
        }
    }

    build_explained_results(scores, provenance, num_retrievers, "dbsf", config.top_k)
}

/// Helper to build explained results from scores and provenance.
fn build_explained_results<I: Clone + Eq + Hash>(
    scores: HashMap<I, f32>,
    mut provenance: HashMap<I, Vec<SourceContribution>>,
    num_retrievers: f32,
    method: &'static str,
    top_k: Option<usize>,
) -> Vec<FusedResult<I>> {
    let mut results: Vec<FusedResult<I>> = scores
        .into_iter()
        .map(|(id, score)| {
            let sources = provenance.remove(&id).unwrap_or_default();
            let consensus_score = sources.len() as f32 / num_retrievers;

            FusedResult {
                id,
                score,
                rank: 0, // Will be set after sorting
                explanation: Explanation {
                    sources,
                    method,
                    consensus_score,
                },
            }
        })
        .collect();

    results.sort_by(|a, b| b.score.total_cmp(&a.score));

    for (rank, result) in results.iter_mut().enumerate() {
        result.rank = rank;
    }

    if let Some(k) = top_k {
        results.truncate(k);
    }

    results
}

// ─────────────────────────────────────────────────────────────────────────────
// Trait-Based Abstraction
// ─────────────────────────────────────────────────────────────────────────────

/// Fusion strategy enum for runtime dispatch.
///
/// This enables dynamic selection of fusion methods without trait objects.
///
/// # Example
///
/// ```rust
/// use rank_fusion::FusionStrategy;
///
/// let strategy = FusionStrategy::rrf(60);
/// let result = strategy.fuse(&[&list1[..], &list2[..]]);
/// ```
#[derive(Debug, Clone)]
pub enum FusionStrategy {
    /// RRF with custom k.
    Rrf { k: u32 },
    /// CombSUM.
    CombSum,
    /// CombMNZ.
    CombMnz,
    /// Weighted fusion with custom weights.
    Weighted { weights: Vec<f32>, normalize: bool },
}

impl FusionStrategy {
    /// Fuse multiple ranked lists.
    ///
    /// # Arguments
    /// * `runs` - Slice of ranked lists, each as (ID, score) pairs
    ///
    /// # Returns
    /// Combined list sorted by fused score (descending)
    pub fn fuse<I: Clone + Eq + Hash>(&self, runs: &[&[(I, f32)]]) -> Vec<(I, f32)> {
        match self {
            Self::Rrf { k } => rrf_multi(runs, RrfConfig::new(*k)),
            Self::CombSum => combsum_multi(runs, FusionConfig::default()),
            Self::CombMnz => combmnz_multi(runs, FusionConfig::default()),
            Self::Weighted { weights, normalize } => {
                if runs.len() != weights.len() {
                    return Vec::new();
                }
                let lists: Vec<_> = runs
                    .iter()
                    .zip(weights.iter())
                    .map(|(run, &w)| (*run, w))
                    .collect();
                weighted_multi(&lists, *normalize, None).unwrap_or_default()
            }
        }
    }

    /// Human-readable name of this fusion method.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Rrf { .. } => "rrf",
            Self::CombSum => "combsum",
            Self::CombMnz => "combmnz",
            Self::Weighted { .. } => "weighted",
        }
    }

    /// Whether this method uses score values (true) or only ranks (false).
    pub fn uses_scores(&self) -> bool {
        match self {
            Self::Rrf { .. } => false,
            Self::CombSum | Self::CombMnz | Self::Weighted { .. } => true,
        }
    }
}

// Convenience constructors for FusionStrategy
impl FusionStrategy {
    /// Create RRF strategy with custom k.
    #[must_use]
    pub fn rrf(k: u32) -> Self {
        assert!(k >= 1, "k must be >= 1");
        Self::Rrf { k }
    }

    /// Create RRF strategy with default k=60.
    #[must_use]
    pub fn rrf_default() -> Self {
        Self::Rrf { k: 60 }
    }

    /// Create CombSUM strategy.
    #[must_use]
    pub fn combsum() -> Self {
        Self::CombSum
    }

    /// Create CombMNZ strategy.
    #[must_use]
    pub fn combmnz() -> Self {
        Self::CombMnz
    }

    /// Create weighted strategy with custom weights.
    #[must_use]
    pub fn weighted(weights: Vec<f32>, normalize: bool) -> Self {
        Self::Weighted { weights, normalize }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Additional Algorithms
// ─────────────────────────────────────────────────────────────────────────────

/// CombMAX: maximum score across all lists.
///
/// Formula: `score(d) = max(s_r(d))` for all retrievers r containing d.
///
/// Use as a baseline or when you want to favor documents that score highly
/// in at least one retriever.
#[must_use]
pub fn combmax<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmax_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombMAX for 3+ result lists.
#[must_use]
pub fn combmax_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            scores
                .entry(id.clone())
                .and_modify(|max_score| *max_score = max_score.max(*s))
                .or_insert(*s);
        }
    }

    finalize(scores, config.top_k)
}

/// CombMED: median score across all lists.
///
/// Formula: `score(d) = median(s_r(d))` for all retrievers r containing d.
///
/// More robust to outliers than CombMAX or CombSUM.
#[must_use]
pub fn combmed<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combmed_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombMED for 3+ result lists.
#[must_use]
pub fn combmed_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut score_lists: HashMap<I, Vec<f32>> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            score_lists.entry(id.clone()).or_default().push(*s);
        }
    }

    let mut scores: HashMap<I, f32> = HashMap::new();
    for (id, mut score_vec) in score_lists {
        score_vec.sort_by(|a, b| a.total_cmp(b));
        let median = if score_vec.len() % 2 == 0 {
            let mid = score_vec.len() / 2;
            (score_vec[mid - 1] + score_vec[mid]) / 2.0
        } else {
            score_vec[score_vec.len() / 2]
        };
        scores.insert(id, median);
    }

    finalize(scores, config.top_k)
}

/// CombANZ: average of non-zero scores.
///
/// Formula: `score(d) = mean(s_r(d))` for all retrievers r containing d.
///
/// Similar to CombSUM but divides by count (average instead of sum).
#[must_use]
pub fn combanz<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    combanz_multi(&[results_a, results_b], FusionConfig::default())
}

/// CombANZ for 3+ result lists.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn combanz_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }
    let mut scores: HashMap<I, (f32, usize)> = HashMap::new();

    for list in lists {
        for (id, s) in list.as_ref() {
            let entry = scores.entry(id.clone()).or_insert((0.0, 0));
            entry.0 += s;
            entry.1 += 1;
        }
    }

    let mut results: Vec<_> = scores
        .into_iter()
        .map(|(id, (sum, count))| (id, sum / count as f32))
        .collect();
    sort_scored_desc(&mut results);
    if let Some(top_k) = config.top_k {
        results.truncate(top_k);
    }
    results
}

/// Rank-Biased Centroids (RBC) fusion.
///
/// Handles variable-length lists gracefully by using a geometric discount
/// that depends on list length. More robust than RRF when lists have very
/// different lengths.
///
/// Formula: `score(d) = Σ (1 - p)^rank / (1 - p^N)` where:
/// - `p` is the persistence parameter (default 0.8, higher = more top-heavy)
/// - `N` is the list length
/// - `rank` is 0-indexed
///
/// From Bailey et al. (2017). Better than RRF when lists have different lengths.
#[must_use]
pub fn rbc<I: Clone + Eq + Hash>(results_a: &[(I, f32)], results_b: &[(I, f32)]) -> Vec<(I, f32)> {
    rbc_multi(&[results_a, results_b], 0.8)
}

/// RBC for 3+ result lists with custom persistence.
///
/// # Arguments
/// * `lists` - Ranked lists to fuse
/// * `persistence` - Persistence parameter (0.0-1.0), default 0.8. Higher = more top-heavy.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn rbc_multi<I, L>(lists: &[L], persistence: f32) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    let p = persistence.clamp(0.0, 1.0);
    let mut scores: HashMap<I, f32> = HashMap::new();

    for list in lists {
        let items = list.as_ref();
        let n = items.len() as f32;
        let denominator = 1.0 - p.powi(n as i32);

        for (rank, (id, _)) in items.iter().enumerate() {
            let numerator = (1.0 - p).powi(rank as i32);
            let contribution = if denominator > 1e-9 {
                numerator / denominator
            } else {
                0.0
            };

            *scores.entry(id.clone()).or_insert(0.0) += contribution;
        }
    }

    finalize(scores, None)
}

/// Condorcet fusion (pairwise comparison voting).
///
/// For each pair of documents, counts how many retrievers prefer one over the other.
/// Documents that beat all others in pairwise comparisons win.
///
/// This is a simplified Condorcet method. Full Condorcet (Kemeny optimal) is NP-hard.
///
/// # Algorithm
///
/// 1. For each document pair (d1, d2), count retrievers where d1 ranks higher than d2
/// 2. Document d1 "beats" d2 if majority of retrievers prefer d1
/// 3. Score = number of documents that this document beats
///
/// More robust to outliers than score-based methods.
#[must_use]
pub fn condorcet<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    condorcet_multi(&[results_a, results_b], FusionConfig::default())
}

/// Condorcet for 3+ result lists.
#[must_use]
pub fn condorcet_multi<I, L>(lists: &[L], config: FusionConfig) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    // Build rank maps: doc_id -> rank in each list
    let mut doc_ranks: HashMap<I, Vec<Option<usize>>> = HashMap::new();
    let mut all_docs: std::collections::HashSet<I> = std::collections::HashSet::new();

    for list in lists {
        let items = list.as_ref();
        for doc_id in items.iter().map(|(id, _)| id) {
            all_docs.insert(doc_id.clone());
        }
    }

    // Initialize all docs with None ranks
    for doc_id in &all_docs {
        doc_ranks.insert(doc_id.clone(), vec![None; lists.len()]);
    }

    // Fill in actual ranks
    for (list_idx, list) in lists.iter().enumerate() {
        for (rank, (id, _)) in list.as_ref().iter().enumerate() {
            if let Some(ranks) = doc_ranks.get_mut(id) {
                ranks[list_idx] = Some(rank);
            }
        }
    }

    // For each document, count how many others it beats
    let mut scores: HashMap<I, f32> = HashMap::new();
    let doc_vec: Vec<I> = all_docs.into_iter().collect();

    for (i, d1) in doc_vec.iter().enumerate() {
        let mut wins = 0;

        for (j, d2) in doc_vec.iter().enumerate() {
            if i == j {
                continue;
            }

            // Count lists where d1 ranks better than d2
            let d1_ranks = &doc_ranks[d1];
            let d2_ranks = &doc_ranks[d2];

            let mut d1_wins = 0;
            for (r1, r2) in d1_ranks.iter().zip(d2_ranks.iter()) {
                match (r1, r2) {
                    (Some(rank1), Some(rank2)) if rank1 < rank2 => d1_wins += 1,
                    (Some(_), None) => d1_wins += 1, // d1 present, d2 not
                    _ => {}
                }
            }

            // Majority wins
            if d1_wins > lists.len() / 2 {
                wins += 1;
            }
        }

        scores.insert(d1.clone(), wins as f32);
    }

    finalize(scores, config.top_k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Optimization and Metrics
// ─────────────────────────────────────────────────────────────────────────────

/// Relevance judgments (qrels) for a query.
///
/// Maps document IDs to relevance scores (typically 0=not relevant, 1=relevant, 2=highly relevant).
pub type Qrels<K> = std::collections::HashMap<K, u32>;

/// Normalized Discounted Cumulative Gain at k.
///
/// Measures ranking quality by rewarding relevant documents that appear early.
/// NDCG@k ranges from 0.0 (worst) to 1.0 (perfect).
///
/// # Formula
///
/// NDCG@k = DCG@k / IDCG@k
///
/// where:
/// - DCG@k = Σ (2^rel_i - 1) / log2(i + 1) for i in [0, k)
/// - IDCG@k = DCG@k of the ideal ranking (sorted by relevance descending)
pub fn ndcg_at_k<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    if qrels.is_empty() || results.is_empty() {
        return 0.0;
    }

    let k = k.min(results.len());
    let mut dcg = 0.0;

    for (i, (id, _)) in results.iter().take(k).enumerate() {
        if let Some(&rel) = qrels.get(id) {
            let gain = (2.0_f32.powi(rel as i32) - 1.0) / ((i + 2) as f32).log2();
            dcg += gain;
        }
    }

    // Compute IDCG (ideal DCG)
    let mut ideal_relevances: Vec<u32> = qrels.values().copied().collect();
    ideal_relevances.sort_by(|a, b| b.cmp(a)); // Descending

    let mut idcg = 0.0;
    for (i, &rel) in ideal_relevances.iter().take(k).enumerate() {
        let gain = (2.0_f32.powi(rel as i32) - 1.0) / ((i + 2) as f32).log2();
        idcg += gain;
    }

    if idcg > 1e-9 {
        dcg / idcg
    } else {
        0.0
    }
}

/// Mean Reciprocal Rank.
///
/// Measures the rank of the first relevant document. MRR ranges from 0.0 to 1.0.
///
/// Formula: MRR = 1 / rank_of_first_relevant
pub fn mrr<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>) -> f32 {
    for (rank, (id, _)) in results.iter().enumerate() {
        if qrels.contains_key(id) && qrels[id] > 0 {
            return 1.0 / (rank + 1) as f32;
        }
    }
    0.0
}

/// Recall at k.
///
/// Fraction of relevant documents that appear in the top-k results.
///
/// Formula: Recall@k = |relevant_docs_in_top_k| / |total_relevant_docs|
pub fn recall_at_k<K: Clone + Eq + Hash>(results: &[(K, f32)], qrels: &Qrels<K>, k: usize) -> f32 {
    let total_relevant = qrels.values().filter(|&&rel| rel > 0).count();
    if total_relevant == 0 {
        return 0.0;
    }

    let k = k.min(results.len());
    let relevant_in_top_k = results
        .iter()
        .take(k)
        .filter(|(id, _)| qrels.get(id).is_some_and(|&rel| rel > 0))
        .count();

    relevant_in_top_k as f32 / total_relevant as f32
}

/// Optimization configuration for hyperparameter search.
#[derive(Debug, Clone)]
pub struct OptimizeConfig {
    /// Fusion method to optimize.
    pub method: FusionMethod,
    /// Metric to optimize (NDCG, MRR, or Recall).
    pub metric: OptimizeMetric,
    /// Parameter grid to search.
    pub param_grid: ParamGrid,
}

/// Metric to optimize during hyperparameter search.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OptimizeMetric {
    /// NDCG@k (default k=10).
    Ndcg { k: usize },
    /// Mean Reciprocal Rank.
    Mrr,
    /// Recall@k (default k=10).
    Recall { k: usize },
}

impl Default for OptimizeMetric {
    fn default() -> Self {
        Self::Ndcg { k: 10 }
    }
}

/// Parameter grid for optimization.
#[derive(Debug, Clone)]
pub enum ParamGrid {
    /// Grid search over RRF k values.
    RrfK { values: Vec<u32> },
    /// Grid search over weighted fusion weights.
    Weighted { weight_combinations: Vec<Vec<f32>> },
}

/// Optimized parameters from hyperparameter search.
#[derive(Debug, Clone)]
pub struct OptimizedParams {
    /// Best metric value found.
    pub best_score: f32,
    /// Parameters that achieved best score.
    pub best_params: String,
}

/// Optimize fusion hyperparameters using grid search.
///
/// Given relevance judgments (qrels) and multiple retrieval runs, searches
/// over parameter space to find the best configuration.
///
/// # Example
///
/// ```rust
/// use rank_fusion::optimize::{optimize_fusion, OptimizeConfig, OptimizeMetric, ParamGrid};
/// use rank_fusion::FusionMethod;
///
/// let qrels = std::collections::HashMap::from([
///     ("doc1", 2), // highly relevant
///     ("doc2", 1), // relevant
/// ]);
///
/// let runs = vec![
///     vec![("doc1", 0.9), ("doc2", 0.8)],
///     vec![("doc2", 0.9), ("doc1", 0.7)],
/// ];
///
/// let config = OptimizeConfig {
///     method: FusionMethod::Rrf { k: 60 }, // will be overridden
///     metric: OptimizeMetric::Ndcg { k: 10 },
///     param_grid: ParamGrid::RrfK {
///         values: vec![20, 40, 60, 100],
///     },
/// };
///
/// let optimized = optimize_fusion(&qrels, &runs, config);
/// println!("Best k: {}, score: {:.4}", optimized.best_params, optimized.best_score);
/// ```
pub fn optimize_fusion<K: Clone + Eq + Hash>(
    qrels: &Qrels<K>,
    runs: &[Vec<(K, f32)>],
    config: OptimizeConfig,
) -> OptimizedParams {
    let mut best_score = f32::NEG_INFINITY;
    let mut best_params = String::new();

    match config.param_grid {
        ParamGrid::RrfK { values } => {
            for k in values {
                let method = FusionMethod::Rrf { k };
                let fused = method.fuse_multi(runs);

                let score = match config.metric {
                    OptimizeMetric::Ndcg { k: ndcg_k } => ndcg_at_k(&fused, qrels, ndcg_k),
                    OptimizeMetric::Mrr => mrr(&fused, qrels),
                    OptimizeMetric::Recall { k: recall_k } => recall_at_k(&fused, qrels, recall_k),
                };

                if score > best_score {
                    best_score = score;
                    best_params = format!("k={}", k);
                }
            }
        }
        ParamGrid::Weighted {
            ref weight_combinations,
        } => {
            for weights in weight_combinations {
                if weights.len() != runs.len() {
                    continue;
                }
                let lists: Vec<(&[(K, f32)], f32)> = runs
                    .iter()
                    .zip(weights.iter())
                    .map(|(run, &w)| (run.as_slice(), w))
                    .collect();

                if let Ok(fused) = weighted_multi(&lists, true, None) {
                    let score = match config.metric {
                        OptimizeMetric::Ndcg { k: ndcg_k } => ndcg_at_k(&fused, qrels, ndcg_k),
                        OptimizeMetric::Mrr => mrr(&fused, qrels),
                        OptimizeMetric::Recall { k: recall_k } => {
                            recall_at_k(&fused, qrels, recall_k)
                        }
                    };

                    if score > best_score {
                        best_score = score;
                        best_params = format!("weights={:?}", weights);
                    }
                }
            }
        }
    }

    OptimizedParams {
        best_score,
        best_params,
    }
}

/// Optimization module exports.
pub mod optimize {
    pub use crate::{
        mrr, ndcg_at_k, optimize_fusion, recall_at_k, OptimizeConfig, OptimizeMetric,
        OptimizedParams, ParamGrid, Qrels,
    };
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
        let f = rrf(&a, &b);

        assert!(f.iter().position(|(id, _)| *id == "d2").unwrap() < 2);
    }

    #[test]
    fn rrf_with_top_k() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(2));

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
        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

        let expected = 1.0 / 60.0;
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    /// Verify RRF score formula: score(d) = Σ 1/(k + rank) for all lists containing d
    #[test]
    fn rrf_exact_score_computation() {
        // d1 at rank 0 in list A, rank 2 in list B
        // With k=60: score = 1/(60+0) + 1/(60+2) = 1/60 + 1/62
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7)];

        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

        // Find d1's score
        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 1.0 / 60.0 + 1.0 / 62.0; // rank 0 in A + rank 2 in B

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    /// Verify ISR score formula: score(d) = Σ 1/sqrt(k + rank)
    #[test]
    fn isr_exact_score_computation() {
        // d1 at rank 0 in list A, rank 2 in list B
        // With k=1: score = 1/sqrt(1+0) + 1/sqrt(1+2) = 1 + 1/sqrt(3)
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7)];

        let f = isr_with_config(&a, &b, RrfConfig::new(1));

        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 1.0 / 1.0_f32.sqrt() + 1.0 / 3.0_f32.sqrt();

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    /// Verify Borda score formula: score(d) = Σ (N - rank) where N = list length
    #[test]
    fn borda_exact_score_computation() {
        // List A: 3 items, d1 at rank 0 -> score = 3-0 = 3
        // List B: 4 items, d1 at rank 2 -> score = 4-2 = 2
        // Total d1 score = 3 + 2 = 5
        let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
        let b = vec![("d4", 0.9), ("d5", 0.8), ("d1", 0.7), ("d6", 0.6)];

        let f = borda(&a, &b);

        let d1_score = f.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let expected = 3.0 + 2.0; // (3-0) + (4-2)

        assert!(
            (d1_score - expected).abs() < 1e-6,
            "d1 score {} != expected {}",
            d1_score,
            expected
        );
    }

    #[test]
    fn rrf_weighted_applies_weights() {
        // d1 appears in list_a (rank 0), d2 appears in list_b (rank 0)
        let list_a = [("d1", 0.0)];
        let list_b = [("d2", 0.0)];

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
        let list_a = [("d1", 0.0)];
        let list_b = [("d2", 0.0)];
        let weights = [0.0, 0.0];

        let result = rrf_weighted(&[&list_a[..], &list_b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::ZeroWeights)));
    }

    // ─────────────────────────────────────────────────────────────────────────
    // ISR Tests
    // ─────────────────────────────────────────────────────────────────────────

    #[test]
    fn isr_basic() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = isr(&a, &b);

        // d2 appears in both lists, should rank high
        assert!(f.iter().position(|(id, _)| *id == "d2").unwrap() < 2);
    }

    #[test]
    fn isr_score_formula() {
        // Single item in one list: score = 1/sqrt(k + 0) = 1/sqrt(k)
        let a = vec![("d1", 1.0)];
        let b: Vec<(&str, f32)> = vec![];
        let f = isr_with_config(&a, &b, RrfConfig::new(1));

        let expected = 1.0 / 1.0_f32.sqrt(); // 1/sqrt(1) = 1.0
        assert!((f[0].1 - expected).abs() < 1e-6);
    }

    #[test]
    fn isr_gentler_decay_than_rrf() {
        // ISR should have a gentler decay than RRF
        // At rank 0 and rank 3 (with k=1):
        // RRF: 1/1 vs 1/4 = ratio of 4
        // ISR: 1/sqrt(1) vs 1/sqrt(4) = 1 vs 0.5 = ratio of 2
        let a = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8), ("d4", 0.7)];
        let b: Vec<(&str, f32)> = vec![];

        let rrf_result = rrf_with_config(&a, &b, RrfConfig::new(1));
        let isr_result = isr_with_config(&a, &b, RrfConfig::new(1));

        // Calculate ratio of first to last score
        let rrf_ratio = rrf_result[0].1 / rrf_result[3].1;
        let isr_ratio = isr_result[0].1 / isr_result[3].1;

        // ISR should have smaller ratio (gentler decay)
        assert!(
            isr_ratio < rrf_ratio,
            "ISR should have gentler decay: ISR ratio={}, RRF ratio={}",
            isr_ratio,
            rrf_ratio
        );
    }

    #[test]
    fn isr_multi_works() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let c = ranked(&["d3", "d4"]);
        let f = isr_multi(&[&a, &b, &c], RrfConfig::new(1));

        // All items should be present
        assert_eq!(f.len(), 4);
        // d2 and d3 appear in 2 lists each, d1 and d4 in 1
        // d2 at rank 1,0 => 1/sqrt(2) + 1/sqrt(1)
        // d3 at rank 1,0 => 1/sqrt(2) + 1/sqrt(1)
        // They should be top
        let top_2: Vec<_> = f.iter().take(2).map(|(id, _)| *id).collect();
        assert!(top_2.contains(&"d2") && top_2.contains(&"d3"));
    }

    #[test]
    fn isr_with_top_k() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d2", "d3", "d4"]);
        let f = isr_with_config(&a, &b, RrfConfig::new(1).with_top_k(2));

        assert_eq!(f.len(), 2);
    }

    #[test]
    fn isr_empty_lists() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = ranked(&["d1"]);

        assert_eq!(isr(&empty, &non_empty).len(), 1);
        assert_eq!(isr(&non_empty, &empty).len(), 1);
        assert_eq!(isr(&empty, &empty).len(), 0);
    }

    #[test]
    fn fusion_method_isr() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = FusionMethod::isr().fuse(&a, &b);
        assert_eq!(f[0].0, "d2");

        // With custom k
        let f = FusionMethod::isr_with_k(10).fuse(&a, &b);
        assert_eq!(f[0].0, "d2");
    }

    #[test]
    fn fusion_method_isr_multi() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);
        let c = ranked(&["d3", "d4"]);
        let lists = [&a[..], &b[..], &c[..]];

        let f = FusionMethod::isr().fuse_multi(&lists);
        assert!(!f.is_empty());
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

        assert_eq!(rrf(&empty, &non_empty).len(), 1);
        assert_eq!(rrf(&non_empty, &empty).len(), 1);
    }

    #[test]
    fn both_empty() {
        let empty: Vec<(&str, f32)> = vec![];
        assert_eq!(rrf(&empty, &empty).len(), 0);
        assert_eq!(combsum(&empty, &empty).len(), 0);
        assert_eq!(borda(&empty, &empty).len(), 0);
    }

    #[test]
    fn duplicate_ids_in_same_list() {
        let a = vec![("d1", 1.0), ("d1", 0.5)];
        let b: Vec<(&str, f32)> = vec![];
        let f = rrf_with_config(&a, &b, RrfConfig::new(60));

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
        let _ = rrf(&a, &b);
        let _ = combsum(&a, &b);
        let _ = combmnz(&a, &b);
        let _ = borda(&a, &b);
    }

    #[test]
    fn inf_scores_handled() {
        let a = vec![("d1", f32::INFINITY), ("d2", 0.5)];
        let b = vec![("d2", f32::NEG_INFINITY), ("d3", 0.1)];

        // Should not panic
        let _ = rrf(&a, &b);
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
        let f = rrf_with_config(&a, &b, RrfConfig::new(u32::MAX));
        assert!(!f.is_empty());
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn k_zero_panics() {
        let _ = RrfConfig::new(0);
    }

    #[test]
    #[should_panic(expected = "k must be >= 1")]
    fn k_zero_with_k_panics() {
        let _ = RrfConfig::default().with_k(0);
    }

    #[test]
    fn all_nan_scores() {
        let a = vec![("d1", f32::NAN), ("d2", f32::NAN)];
        let b = vec![("d3", f32::NAN), ("d4", f32::NAN)];

        // Should not panic, but results may contain NaN
        let f = rrf(&a, &b);
        assert_eq!(f.len(), 4);
        // NaN values are valid RRF scores (1/(k+rank) is always finite)
        // But if all scores are NaN, the RRF calculation still works
        // Actually, RRF ignores scores, so NaN scores don't matter
        // All documents get RRF scores based on ranks, which are finite
        for (_, score) in &f {
            assert!(
                score.is_finite(),
                "RRF scores should be finite (based on ranks, not input scores)"
            );
        }
    }

    #[test]
    fn empty_lists_multi() {
        let empty: Vec<Vec<(&str, f32)>> = vec![];
        assert_eq!(rrf_multi(&empty, RrfConfig::default()).len(), 0);
        assert_eq!(combsum_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(combmnz_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(borda_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(dbsf_multi(&empty, FusionConfig::default()).len(), 0);
        assert_eq!(isr_multi(&empty, RrfConfig::default()).len(), 0);
    }

    #[test]
    fn rrf_weighted_list_weight_mismatch() {
        let a = [("d1", 1.0)];
        let b = [("d2", 1.0)];
        let weights = [0.5, 0.5, 0.0]; // 3 weights for 2 lists

        let result = rrf_weighted(&[&a[..], &b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn rrf_weighted_list_weight_mismatch_short() {
        let a = [("d1", 1.0)];
        let b = [("d2", 1.0)];
        let weights = [0.5]; // 1 weight for 2 lists

        let result = rrf_weighted(&[&a[..], &b[..]], &weights, RrfConfig::default());
        assert!(matches!(result, Err(FusionError::InvalidConfig(_))));
    }

    #[test]
    fn duplicate_ids_commutative() {
        // Test that duplicate handling is commutative
        let a = vec![("d1", 1.0), ("d1", 0.5), ("d2", 0.3)];
        let b = vec![("d2", 0.9), ("d3", 0.7)];

        let ab = rrf(&a, &b);
        let ba = rrf(&b, &a);

        // Should have same document IDs (order may differ due to ties)
        let ab_ids: Vec<&str> = ab.iter().map(|(id, _)| *id).collect();
        let ba_ids: Vec<&str> = ba.iter().map(|(id, _)| *id).collect();
        assert_eq!(ab_ids.len(), ba_ids.len());
        // All IDs should appear in both
        for id in &ab_ids {
            assert!(ba_ids.contains(id));
        }
    }

    #[test]
    fn dbsf_zero_variance() {
        // All scores equal in one list
        let a = vec![("d1", 1.0), ("d2", 1.0), ("d3", 1.0)];
        let b = vec![("d1", 0.9), ("d2", 0.5), ("d3", 0.1)];

        // Should not panic, list a contributes z-score=0.0 for all
        let f = dbsf(&a, &b);
        assert_eq!(f.len(), 3);
        // d1 should win (0.0 + positive z-score from b)
        assert_eq!(f[0].0, "d1");
    }

    #[test]
    fn single_item_lists() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d1", 1.0)];

        let f = rrf(&a, &b);
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

        let f = rrf(&a, &b);
        assert_eq!(f.len(), 4);

        let f = combmnz(&a, &b);
        assert_eq!(f.len(), 4);
        // No overlap bonus
    }

    #[test]
    fn identical_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d1", "d2", "d3"]);

        let f = rrf(&a, &b);
        // Order should be preserved
        assert_eq!(f[0].0, "d1");
        assert_eq!(f[1].0, "d2");
        assert_eq!(f[2].0, "d3");
    }

    #[test]
    fn reversed_lists() {
        let a = ranked(&["d1", "d2", "d3"]);
        let b = ranked(&["d3", "d2", "d1"]);

        let f = rrf(&a, &b);
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

        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(100));
        assert_eq!(f.len(), 2);
    }

    #[test]
    fn top_k_zero() {
        let a = ranked(&["d1", "d2"]);
        let b = ranked(&["d2", "d3"]);

        let f = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(0));
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

    #[test]
    fn rrf_multi_empty_lists() {
        let empty: Vec<Vec<(u32, f32)>> = vec![];
        let result = rrf_multi(&empty, RrfConfig::default());
        assert!(result.is_empty());
    }

    proptest! {
        #[test]
        fn rrf_output_bounded(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(&a, &b);
            prop_assert!(result.len() <= a.len() + b.len());
        }

        #[test]
        fn rrf_scores_positive(a in arb_results(50), b in arb_results(50)) {
            let result = rrf(&a, &b);
            for (_, score) in &result {
                prop_assert!(*score > 0.0);
            }
        }

        #[test]
        fn rrf_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = rrf(&a, &b);
            let ba = rrf(&b, &a);

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
            let result = rrf(&a, &b);
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        #[test]
        fn rrf_top_k_respected(a in arb_results(50), b in arb_results(50), k in 1usize..20) {
            let result = rrf_with_config(&a, &b, RrfConfig::default().with_top_k(k));
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

        /// CombMNZ should be commutative (argument order doesn't change scores)
        #[test]
        fn combmnz_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = combmnz(&a, &b);
            let ba = combmnz(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5,
                    "CombMNZ not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        /// DBSF should be commutative (argument order doesn't change normalized scores)
        #[test]
        fn dbsf_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = dbsf(&a, &b);
            let ba = dbsf(&b, &a);

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            prop_assert_eq!(ab_map.len(), ba_map.len());
            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-5,
                    "DBSF not commutative for id {:?}: {} vs {}", id, score_ab, score_ba);
            }
        }

        #[test]
        fn rrf_duplicate_handling_commutative(a in arb_results(20), b in arb_results(20)) {
            // Create lists with potential duplicates by allowing same IDs
            let mut a_with_dups = a.clone();
            // Add a duplicate at a different rank
            if !a_with_dups.is_empty() {
                let dup_id = a_with_dups[0].0;
                a_with_dups.push((dup_id, 0.5));
            }

            let ab = rrf(&a_with_dups, &b);
            let ba = rrf(&b, &a_with_dups);

            // Should have same document IDs (order may differ due to ties)
            let ab_ids: std::collections::HashSet<_> = ab.iter().map(|(id, _)| *id).collect();
            let ba_ids: std::collections::HashSet<_> = ba.iter().map(|(id, _)| *id).collect();
            prop_assert_eq!(ab_ids, ba_ids);
        }

        #[test]
        fn rrf_multi_some_empty_lists(
            a in arb_results(10).prop_filter("need items", |v| !v.is_empty()),
            b in arb_results(10).prop_filter("need items", |v| !v.is_empty())
        ) {
            let empty: Vec<(u32, f32)> = vec![];
            let lists: Vec<&[(u32, f32)]> = vec![&a, &empty, &b];
            let result = rrf_multi(&lists, RrfConfig::default());
            // Should still produce results from non-empty lists
            prop_assert!(!result.is_empty());
        }

        #[test]
        fn rrf_weighted_length_validation(
            a in arb_results(5),
            b in arb_results(5)
        ) {
            // Test that mismatched lengths are caught
            let lists: Vec<&[(u32, f32)]> = vec![&a, &b];
            let weights_short = [0.5]; // 1 weight for 2 lists
            let weights_long = [0.3, 0.3, 0.4]; // 3 weights for 2 lists

            let result_short = rrf_weighted(&lists, &weights_short, RrfConfig::default());
            let result_long = rrf_weighted(&lists, &weights_long, RrfConfig::default());

            prop_assert!(matches!(result_short, Err(FusionError::InvalidConfig(_))));
            prop_assert!(matches!(result_long, Err(FusionError::InvalidConfig(_))));
        }

        #[test]
        fn rrf_k_uniformity(a in arb_results(10).prop_filter("need items", |v| v.len() >= 2)) {
            let b: Vec<(u32, f32)> = vec![];

            let low_k = rrf_with_config(&a, &b, RrfConfig::new(1));
            let high_k = rrf_with_config(&a, &b, RrfConfig::new(1000));

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

            prop_assert!(!rrf(&a, &b).is_empty());
            prop_assert!(!isr(&a, &b).is_empty());
            prop_assert!(!combsum(&a, &b).is_empty());
            prop_assert!(!combmnz(&a, &b).is_empty());
            prop_assert!(!borda(&a, &b).is_empty());
        }

        // ─────────────────────────────────────────────────────────────────────────
        // ISR Property Tests
        // ─────────────────────────────────────────────────────────────────────────

        /// ISR output should be bounded by input size
        #[test]
        fn isr_output_bounded(a in arb_results(50), b in arb_results(50)) {
            let result = isr(&a, &b);
            prop_assert!(result.len() <= a.len() + b.len());
        }

        /// ISR scores should be positive
        #[test]
        fn isr_scores_positive(a in arb_results(50), b in arb_results(50)) {
            let result = isr(&a, &b);
            for (_, score) in &result {
                prop_assert!(*score > 0.0);
            }
        }

        /// ISR should be commutative
        #[test]
        fn isr_commutative(a in arb_results(20), b in arb_results(20)) {
            let ab = isr(&a, &b);
            let ba = isr(&b, &a);

            prop_assert_eq!(ab.len(), ba.len());

            let ab_map: HashMap<_, _> = ab.into_iter().collect();
            let ba_map: HashMap<_, _> = ba.into_iter().collect();

            for (id, score_ab) in &ab_map {
                let score_ba = ba_map.get(id).expect("same keys");
                prop_assert!((score_ab - score_ba).abs() < 1e-6);
            }
        }

        /// ISR should be sorted descending
        #[test]
        fn isr_sorted_descending(a in arb_results(50), b in arb_results(50)) {
            let result = isr(&a, &b);
            for window in result.windows(2) {
                prop_assert!(window[0].1 >= window[1].1);
            }
        }

        /// ISR top_k should be respected
        #[test]
        fn isr_top_k_respected(a in arb_results(50), b in arb_results(50), k in 1usize..20) {
            let result = isr_with_config(&a, &b, RrfConfig::new(1).with_top_k(k));
            prop_assert!(result.len() <= k);
        }

        /// ISR should have gentler decay than RRF (higher relative contribution from lower ranks)
        /// Test uses unique IDs to isolate the decay function comparison
        #[test]
        fn isr_gentler_than_rrf(n in 3usize..20) {
            // Create list with unique IDs at sequential ranks
            let a: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, 1.0)).collect();
            let b: Vec<(u32, f32)> = vec![];

            let rrf_result = rrf_with_config(&a, &b, RrfConfig::new(1));
            let isr_result = isr_with_config(&a, &b, RrfConfig::new(1));

            // Both should have all n unique items
            prop_assert_eq!(rrf_result.len(), n);
            prop_assert_eq!(isr_result.len(), n);

            // Compare ratio of first to last score
            let rrf_ratio = rrf_result[0].1 / rrf_result.last().unwrap().1;
            let isr_ratio = isr_result[0].1 / isr_result.last().unwrap().1;

            // ISR should have smaller ratio (gentler decay)
            // RRF: 1/k vs 1/(k+n-1) => ratio = (k+n-1)/k = n for k=1
            // ISR: 1/sqrt(k) vs 1/sqrt(k+n-1) => ratio = sqrt(k+n-1)/sqrt(k) = sqrt(n) for k=1
            // sqrt(n) < n for n > 1, so ISR ratio should be smaller
            prop_assert!(isr_ratio < rrf_ratio,
                "ISR ratio {} should be < RRF ratio {} for n={}", isr_ratio, rrf_ratio, n);
        }

        /// multi variants should match two-list for n=2 (same scores, may differ in order for ties)
        #[test]
        fn multi_matches_two_list(a in arb_results(10), b in arb_results(10)) {
            let two_list = rrf(&a, &b);
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
                rrf(&a, &b),
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
            let result = rrf(&a, &b);
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

            let result = rrf_with_config(&a, &b, RrfConfig::new(k));
            let max_possible = 2.0 / k as f32; // rank 0 in both lists
            prop_assert!(result[0].1 <= max_possible + 1e-6);
        }

        /// Empty list handling: combining with empty should equal single list
        #[test]
        fn empty_list_preserves_ids(n in 1usize..10) {
            // Create list with unique IDs
            let a: Vec<(u32, f32)> = (0..n as u32).map(|i| (i, 1.0 - i as f32 * 0.1)).collect();
            let empty: Vec<(u32, f32)> = vec![];

            let rrf_result = rrf(&a, &empty);

            // Should have same number of unique IDs
            prop_assert_eq!(rrf_result.len(), n);

            // All original IDs should be present
            for (id, _) in &a {
                prop_assert!(rrf_result.iter().any(|(rid, _)| rid == id), "Missing ID {:?}", id);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Explainability Tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod explain_tests {
    use super::*;

    #[test]
    fn rrf_explain_basic() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(&[&bm25[..], &dense[..]], &retrievers, RrfConfig::default());

        // d2 should be top (appears in both lists)
        assert_eq!(explained[0].id, "d2");
        assert_eq!(explained[0].explanation.sources.len(), 2);
        assert!((explained[0].explanation.consensus_score - 1.0).abs() < 1e-6);

        // d1 and d3 should each have 1 source
        let d1 = explained.iter().find(|r| r.id == "d1").unwrap();
        assert_eq!(d1.explanation.sources.len(), 1);
        assert!((d1.explanation.consensus_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn rrf_explain_provenance() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(&[&bm25[..], &dense[..]], &retrievers, RrfConfig::new(60));

        let d2 = explained.iter().find(|r| r.id == "d2").unwrap();

        // Check BM25 contribution
        let bm25_contrib = d2
            .explanation
            .sources
            .iter()
            .find(|s| s.retriever_id == "bm25")
            .unwrap();
        assert_eq!(bm25_contrib.original_rank, Some(1));
        assert_eq!(bm25_contrib.original_score, Some(11.0));
        let expected_contrib = 1.0 / (60.0 + 1.0);
        assert!((bm25_contrib.contribution - expected_contrib).abs() < 1e-6);

        // Check dense contribution
        let dense_contrib = d2
            .explanation
            .sources
            .iter()
            .find(|s| s.retriever_id == "dense")
            .unwrap();
        assert_eq!(dense_contrib.original_rank, Some(0));
        assert_eq!(dense_contrib.original_score, Some(0.9));
        let expected_contrib = 1.0 / 60.0;
        assert!((dense_contrib.contribution - expected_contrib).abs() < 1e-6);
    }

    #[test]
    fn rrf_explain_ranks() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(&[&bm25[..], &dense[..]], &retrievers, RrfConfig::default());

        // Ranks should be 0-indexed and sequential
        for (rank, result) in explained.iter().enumerate() {
            assert_eq!(result.rank, rank);
        }
    }

    #[test]
    fn analyze_consensus_basic() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(&[&bm25[..], &dense[..]], &retrievers, RrfConfig::default());

        let consensus = analyze_consensus(&explained);

        // d2 appears in both lists (high consensus)
        assert!(consensus.high_consensus.contains(&"d2"));

        // d1 and d3 appear in only one list (single source)
        assert!(consensus.single_source.contains(&"d1"));
        assert!(consensus.single_source.contains(&"d3"));
    }

    #[test]
    fn attribute_top_k_basic() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(&[&bm25[..], &dense[..]], &retrievers, RrfConfig::default());

        let attribution = attribute_top_k(&explained, 3);

        // Both retrievers should have contributed
        assert!(attribution.contains_key("bm25"));
        assert!(attribution.contains_key("dense"));

        let bm25_stats = &attribution["bm25"];
        assert!(bm25_stats.top_k_count > 0);

        let dense_stats = &attribution["dense"];
        assert!(dense_stats.top_k_count > 0);
    }

    #[test]
    fn rrf_explain_empty_lists() {
        let empty: Vec<(&str, f32)> = vec![];
        let non_empty = vec![("d1", 1.0)];

        let retrievers = vec![RetrieverId::new("empty"), RetrieverId::new("non_empty")];

        let explained = rrf_explain(
            &[&empty[..], &non_empty[..]],
            &retrievers,
            RrfConfig::default(),
        );

        assert_eq!(explained.len(), 1);
        assert_eq!(explained[0].id, "d1");
        assert_eq!(explained[0].explanation.sources.len(), 1);
    }

    #[test]
    fn rrf_explain_mismatched_lengths() {
        let bm25 = vec![("d1", 12.5)];
        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        // Mismatch: 1 list but 2 retriever IDs
        let explained = rrf_explain(&[&bm25[..]], &retrievers, RrfConfig::default());

        // Should return empty (safety check)
        assert!(explained.is_empty());
    }

    #[test]
    fn rrf_explain_top_k() {
        let bm25 = vec![("d1", 12.5), ("d2", 11.0), ("d3", 10.0)];
        let dense = vec![("d2", 0.9), ("d3", 0.8), ("d4", 0.7)];

        let retrievers = vec![RetrieverId::new("bm25"), RetrieverId::new("dense")];

        let explained = rrf_explain(
            &[&bm25[..], &dense[..]],
            &retrievers,
            RrfConfig::default().with_top_k(2),
        );

        assert_eq!(explained.len(), 2);
    }

    #[test]
    fn combsum_explain_basic() {
        let a = vec![("d1", 1.0), ("d2", 0.5)];
        let b = vec![("d2", 1.0), ("d3", 0.5)];

        let retrievers = vec![RetrieverId::new("a"), RetrieverId::new("b")];

        let explained = combsum_explain(&[&a[..], &b[..]], &retrievers, FusionConfig::default());

        assert_eq!(explained.len(), 3);
        let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
        assert_eq!(d2.explanation.sources.len(), 2);
    }

    #[test]
    fn combmax_basic() {
        let a = [("d1", 1.0), ("d2", 0.5)];
        let b = [("d2", 0.8), ("d3", 0.9)];

        let result = combmax(&a, &b);
        assert_eq!(result.len(), 3);
        // d2 should have max(0.5, 0.8) = 0.8
        let d2_score = result.iter().find(|(id, _)| *id == "d2").unwrap().1;
        assert!((d2_score - 0.8).abs() < 1e-6);
    }

    #[test]
    fn combmed_basic() {
        let a = [("d1", 1.0), ("d2", 0.5)];
        let b = [("d2", 0.8), ("d3", 0.9)];

        let result = combmed(&a, &b);
        assert_eq!(result.len(), 3);
        // d2 should have median(0.5, 0.8) = 0.65
        let d2_score = result.iter().find(|(id, _)| *id == "d2").unwrap().1;
        assert!((d2_score - 0.65).abs() < 1e-6);
    }

    #[test]
    fn rbc_basic() {
        let a = [("d1", 1.0), ("d2", 0.5)];
        let b = [("d2", 0.8), ("d3", 0.9)];

        let result = rbc(&a, &b);
        assert!(!result.is_empty());
        // d2 appears in both lists, should rank high
        let d2_rank = result.iter().position(|(id, _)| *id == "d2").unwrap();
        assert!(d2_rank < 2);
    }

    #[test]
    fn condorcet_basic() {
        let a = [("d1", 1.0), ("d2", 0.5), ("d3", 0.3)];
        let b = [("d2", 0.9), ("d1", 0.8), ("d3", 0.7)];

        let result = condorcet(&a, &b);
        assert!(!result.is_empty());
        // d2 should win (rank 1,0 vs d1's 0,1)
    }

    #[test]
    fn ndcg_basic() {
        let results = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8)];
        let qrels = std::collections::HashMap::from([
            ("d1", 2), // highly relevant
            ("d2", 1), // relevant
        ]);

        let score = ndcg_at_k(&results, &qrels, 3);
        assert!(score > 0.0 && score <= 1.0);
    }

    #[test]
    fn mrr_basic() {
        let results = vec![("d1", 1.0), ("d2", 0.9)];
        let qrels = std::collections::HashMap::from([("d2", 1)]);

        let score = mrr(&results, &qrels);
        // d2 is at rank 1 (0-indexed), so MRR = 1/2 = 0.5
        assert!((score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn recall_basic() {
        let results = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8)];
        let qrels = std::collections::HashMap::from([
            ("d1", 1),
            ("d2", 1),
            ("d4", 1), // not in results
        ]);

        let score = recall_at_k(&results, &qrels, 3);
        // 2 relevant docs in top-3 out of 3 total = 2/3
        assert!((score - 2.0 / 3.0).abs() < 1e-6);
    }

    #[test]
    fn fusion_strategy_rrf() {
        let strategy = FusionStrategy::rrf(60);
        let a = [("d1", 1.0), ("d2", 0.5)];
        let b = [("d2", 0.9), ("d3", 0.8)];

        let result = strategy.fuse(&[&a[..], &b[..]]);
        assert!(!result.is_empty());
        assert_eq!(strategy.name(), "rrf");
        assert!(!strategy.uses_scores());
    }

    #[test]
    fn normalize_scores_minmax() {
        let scores = vec![("d1", 10.0), ("d2", 5.0), ("d3", 0.0)];
        let normalized = normalize_scores(&scores, Normalization::MinMax);

        assert_eq!(normalized.len(), 3);
        // d1 should be 1.0, d3 should be 0.0
        let d1 = normalized.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let d3 = normalized.iter().find(|(id, _)| *id == "d3").unwrap().1;
        assert!((d1 - 1.0).abs() < 1e-6);
        assert!((d3 - 0.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_scores_zscore() {
        let scores = vec![("d1", 10.0), ("d2", 5.0), ("d3", 0.0)];
        let normalized = normalize_scores(&scores, Normalization::ZScore);

        assert_eq!(normalized.len(), 3);
        // Mean = 5.0, std ≈ 4.08, so d1 should be positive, d3 negative
        let d1 = normalized.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let d3 = normalized.iter().find(|(id, _)| *id == "d3").unwrap().1;
        assert!(d1 > 0.0);
        assert!(d3 < 0.0);
    }

    #[test]
    fn normalize_scores_sum() {
        let scores = vec![("d1", 2.0), ("d2", 1.0), ("d3", 1.0)];
        let normalized = normalize_scores(&scores, Normalization::Sum);

        let sum: f32 = normalized.iter().map(|(_, s)| s).sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn normalize_scores_rank() {
        let scores = vec![("d1", 10.0), ("d2", 5.0), ("d3", 0.0)];
        let normalized = normalize_scores(&scores, Normalization::Rank);

        // Highest score should get highest rank-normalized value
        let d1 = normalized.iter().find(|(id, _)| *id == "d1").unwrap().1;
        let d3 = normalized.iter().find(|(id, _)| *id == "d3").unwrap().1;
        assert!(d1 > d3);
    }

    #[test]
    fn combsum_explain_shows_normalization() {
        let a = vec![("d1", 1.0), ("d2", 0.5)];
        let b = vec![("d2", 1.0), ("d3", 0.5)];

        let retrievers = vec![RetrieverId::new("a"), RetrieverId::new("b")];
        let explained = combsum_explain(&[&a[..], &b[..]], &retrievers, FusionConfig::default());

        let d2 = explained.iter().find(|r| r.id == "d2").unwrap();
        // d2 should have normalized scores from both sources
        assert_eq!(d2.explanation.sources.len(), 2);
        for source in &d2.explanation.sources {
            assert!(source.normalized_score.is_some());
        }
    }

    #[test]
    fn combmnz_explain_shows_overlap_multiplier() {
        let a = vec![("d1", 1.0)];
        let b = vec![("d1", 1.0), ("d2", 0.5)];

        let retrievers = vec![RetrieverId::new("a"), RetrieverId::new("b")];
        let explained = combmnz_explain(&[&a[..], &b[..]], &retrievers, FusionConfig::default());

        let d1 = explained.iter().find(|r| r.id == "d1").unwrap();
        // d1 appears in both lists, so contributions should reflect multiplier
        assert_eq!(d1.explanation.sources.len(), 2);
        // Final score should be higher than combsum due to overlap bonus
        let combsum_result = combsum(&a, &b);
        let combsum_d1 = combsum_result.iter().find(|(id, _)| *id == "d1").unwrap().1;
        assert!(d1.score > combsum_d1);
    }

    #[test]
    fn dbsf_explain_shows_zscore() {
        let a = vec![("d1", 10.0), ("d2", 5.0), ("d3", 0.0)];
        let b = vec![("d1", 0.9), ("d2", 0.5), ("d4", 0.1)];

        let retrievers = vec![RetrieverId::new("a"), RetrieverId::new("b")];
        let explained = dbsf_explain(&[&a[..], &b[..]], &retrievers, FusionConfig::default());

        let d1 = explained.iter().find(|r| r.id == "d1").unwrap();
        // Should have z-score normalized contributions
        for source in &d1.explanation.sources {
            assert!(source.normalized_score.is_some());
            // Z-scores should be in reasonable range (clipped to [-3, 3])
            let z = source.normalized_score.unwrap();
            assert!(z >= -3.0 && z <= 3.0);
        }
    }

    #[test]
    fn optimize_fusion_rrf_k() {
        let qrels = std::collections::HashMap::from([("d1", 2), ("d2", 1)]);

        let run1 = vec![("d1", 0.9), ("d2", 0.8)];
        let run2 = vec![("d2", 0.9), ("d1", 0.7)];
        let runs = vec![run1, run2];

        let config = OptimizeConfig {
            method: FusionMethod::Rrf { k: 60 },
            metric: OptimizeMetric::Ndcg { k: 10 },
            param_grid: ParamGrid::RrfK {
                values: vec![20, 60, 100],
            },
        };

        let optimized = optimize_fusion(&qrels, &runs, config);
        assert!(optimized.best_score >= 0.0 && optimized.best_score <= 1.0);
        assert!(!optimized.best_params.is_empty());
    }

    #[test]
    fn optimize_fusion_weighted() {
        let qrels = std::collections::HashMap::from([("d1", 1)]);

        let run1 = [("d1", 0.9), ("d2", 0.5)];
        let run2 = [("d1", 0.8), ("d2", 0.6)];
        let runs = vec![run1.to_vec(), run2.to_vec()];

        let config = OptimizeConfig {
            method: FusionMethod::Weighted {
                weight_a: 0.5,
                weight_b: 0.5,
                normalize: true,
            },
            metric: OptimizeMetric::Mrr,
            param_grid: ParamGrid::Weighted {
                weight_combinations: vec![vec![0.5, 0.5], vec![0.7, 0.3], vec![0.3, 0.7]],
            },
        };

        let optimized = optimize_fusion(&qrels, &runs, config);
        assert!(optimized.best_score >= 0.0);
    }

    #[test]
    fn metrics_ndcg_perfect_ranking() {
        // Perfect ranking: relevant docs at top
        let results = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8)];
        let qrels = std::collections::HashMap::from([
            ("d1", 2), // highly relevant
            ("d2", 1), // relevant
        ]);

        let ndcg = ndcg_at_k(&results, &qrels, 10);
        assert!(ndcg > 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn metrics_mrr_first_relevant() {
        let results = vec![("d1", 1.0), ("d2", 0.9)];
        let qrels = std::collections::HashMap::from([("d2", 1)]);

        let mrr_score = mrr(&results, &qrels);
        // d2 is at rank 1 (0-indexed), so MRR = 1/2 = 0.5
        assert!((mrr_score - 0.5).abs() < 1e-6);
    }

    #[test]
    fn metrics_recall_complete() {
        let results = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8)];
        let qrels = std::collections::HashMap::from([
            ("d1", 1),
            ("d2", 1),
            ("d4", 1), // not in results
        ]);

        let recall = recall_at_k(&results, &qrels, 10);
        // 2 relevant docs in results out of 3 total = 2/3
        assert!((recall - 2.0 / 3.0).abs() < 1e-6);
    }
}
