# Concrete Implementation Plans: rank-fusion

## Research-Backed Improvements

These plans are based on recent research papers (2024-2025) with validated improvements:

1. **Standardization-Based Fusion** (ERANK, arXiv:2509.00520)
   - Uses z-score normalization for BM25 hybrid fusion
   - 2-5% NDCG improvement over CombSUM when distributions differ
   - Validated on BRIGHT, TREC DL, BEIR benchmarks

2. **Additive Multi-Task Fusion** (ResFlow, arXiv:2411.09705)
   - Additive fusion outperforms multiplicative: `α·CTR + β·CTCVR` beats `CTR^α × CVR^β`
   - 1.29% OPU increase in online A/B tests at Shopee
   - More robust to extreme values than multiplicative

3. **QPP-Based Routing** (Inspired by Reranker-Guided Search)
   - Automatically detects hard queries and routes to specialized strategies
   - Reduces need for manual tuning

## Implementation Priority

Based on research validation and implementation complexity:
1. Standardized fusion (highest impact, reuses existing code)
2. Additive multi-task fusion (validated in production)
3. QPP routing (requires more validation)

## Plan 1: Standardization-Based Fusion (ERANK-style)

### Overview
Add a new fusion method that uses z-score standardization (mean=0, std=1) instead of min-max normalization, then applies additive fusion. Based on ERANK (arXiv:2509.00520) which uses standardization for BM25 hybrid fusion: `0.2 × normalized_BM25 + 0.8 × normalized_rerank` where normalization is z-score (standardization), not min-max. This shows 2-5% NDCG improvement over CombSUM when score distributions differ significantly.

**Key insight from ERANK**: They apply z-score normalization (standardization) to both BM25 and reranking scores before weighted fusion, finding this more robust than min-max normalization for hybrid scenarios.

### API Design

```rust
// New config struct
#[derive(Debug, Clone)]
pub struct StandardizedConfig {
    /// Clip z-scores to this range (default: [-3.0, 3.0])
    pub clip_range: (f32, f32),
    /// Top-K truncation (None = no truncation)
    pub top_k: Option<usize>,
}

impl Default for StandardizedConfig {
    fn default() -> Self {
        Self {
            clip_range: (-3.0, 3.0),
            top_k: None,
        }
    }
}

// Main API functions
pub fn standardized<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    standardized_with_config(results_a, results_b, StandardizedConfig::default())
}

pub fn standardized_with_config<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    config: StandardizedConfig,
) -> Vec<(I, f32)> {
    standardized_multi(&[results_a, results_b], config)
}

pub fn standardized_multi<I, L>(
    lists: &[L],
    config: StandardizedConfig,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    // Implementation below
}
```

### Algorithm Details

```rust
fn standardized_multi<I, L>(
    lists: &[L],
    config: StandardizedConfig,
) -> Vec<(I, f32)>
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
        if items.is_empty() {
            continue;
        }

        // Compute mean and std for z-score normalization
        // Reuse existing helper from DBSF implementation
        let (mean, std) = zscore_params(items);
        
        // If std is too small, all scores are effectively equal
        // Use min-max as fallback to avoid division by zero
        if std < SCORE_RANGE_EPSILON {
            let (norm, off) = min_max_params(items);
            for (id, s) in items {
                let contribution = (s - off) * norm;
                *scores.entry(id.clone()).or_insert(0.0) += contribution;
            }
        } else {
            let (clip_min, clip_max) = config.clip_range;
            for (id, s) in items {
                // Z-score: (score - mean) / std
                let z = (s - mean) / std;
                // Clip to prevent outliers from dominating (ERANK uses [-3, 3])
                let clipped_z = z.clamp(clip_min, clip_max);
                *scores.entry(id.clone()).or_insert(0.0) += clipped_z;
            }
        }
    }

    finalize(scores, config.top_k)
}
```

**Note**: ERANK uses standardization specifically for BM25 hybrid fusion with weights `0.2 × normalized_BM25 + 0.8 × normalized_rerank`. This is more robust than min-max when score distributions differ significantly (BM25: 0-100 scale, dense: 0-1 scale).

### Integration Points

1. **Add to `FusionMethod` enum** (line ~2050):
```rust
pub enum FusionMethod {
    // ... existing variants ...
    Standardized(StandardizedConfig),
    AdditiveMultiTask(AdditiveMultiTaskConfig),
}
```

2. **Add builder methods**:
```rust
impl FusionMethod {
    pub fn standardized(clip_range: (f32, f32)) -> Self {
        Self::Standardized(StandardizedConfig {
            clip_range,
            top_k: None,
        })
    }
    
    pub fn additive_multi_task(weights: (f32, f32)) -> Self {
        Self::AdditiveMultiTask(AdditiveMultiTaskConfig::new(weights))
    }
}
```

3. **Update `FusionBuilder`** to support both standardized and additive multi-task fusion.

4. **Reuse existing helpers**:
   - `zscore_params()` already exists (used by DBSF)
   - `min_max_params()` already exists (used by CombSUM)
   - `finalize()` already exists for sorting and truncation

### Testing Strategy

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_standardized_basic() {
        let a = vec![("d1", 10.0), ("d2", 5.0), ("d3", 1.0)];
        let b = vec![("d2", 0.9), ("d3", 0.7), ("d4", 0.5)];
        let fused = standardized(&a, &b);
        
        // d2 appears in both, should rank highest
        assert_eq!(fused[0].0, "d2");
    }

    #[test]
    fn test_standardized_zero_variance() {
        // All scores equal in one list
        let a = vec![("d1", 5.0), ("d2", 5.0), ("d3", 5.0)];
        let b = vec![("d1", 0.9), ("d2", 0.7)];
        let fused = standardized(&a, &b);
        
        // Should fallback to min-max for list A
        assert!(!fused.is_empty());
    }

    #[test]
    fn test_standardized_clipping() {
        let config = StandardizedConfig {
            clip_range: (-1.0, 1.0),
            top_k: None,
        };
        let a = vec![("d1", 100.0), ("d2", 1.0)]; // d1 is extreme outlier
        let b = vec![("d1", 0.5), ("d2", 0.9)];
        let fused = standardized_with_config(&a, &b, config);
        
        // Clipping should prevent d1 from dominating
        // d2 should rank higher due to better consensus
    }

    #[test]
    fn test_standardized_vs_combsum() {
        // Test on synthetic data where distributions differ
        // Standardized should handle this better
    }
}
```

### Performance Considerations

- Reuse existing `zscore_params` helper (already exists for DBSF)
- Same complexity as CombSUM: O(L×N + U×log U)
- No additional allocations beyond existing fusion methods
- Z-score computation: O(N) per list for mean/std, same as min-max

### Edge Cases (from ERANK paper)

- **Zero variance**: Falls back to min-max normalization (handled in algorithm)
- **Outlier clipping**: ERANK uses [-3, 3] clipping to prevent extreme values from dominating
- **Empty lists**: Handled by early return (same as other fusion methods)
- **Single-element lists**: Z-score undefined, falls back to min-max (returns norm=1, offset=0)

### Documentation

Add to main lib.rs explaining:
- When to use: Score distributions differ significantly (BM25 vs dense)
- Trade-off: More robust to outliers than CombSUM, but slightly slower
- Empirical results: 2-5% NDCG improvement in ERANK paper

---

## Plan 2: Query Performance Prediction (QPP) Routing

### Overview
Automatically detect "hard" queries and route them to specialized fusion strategies. Based on Reranker-Guided Search (arXiv:2405.07519) which uses QPP to identify queries requiring reasoning.

### API Design

```rust
/// Query difficulty estimator.
///
/// Predicts query hardness based on result list characteristics.
#[derive(Debug, Clone)]
pub struct QueryDifficulty {
    /// Estimated difficulty (0.0 = easy, 1.0 = hard)
    pub score: f32,
    /// Reason for difficulty classification
    pub reason: DifficultyReason,
}

#[derive(Debug, Clone, PartialEq)]
pub enum DifficultyReason {
    /// Low overlap between result lists (disagreement)
    LowOverlap { overlap_ratio: f32 },
    /// High variance in scores (uncertainty)
    HighVariance { cv: f32 },
    /// Shallow result lists (few candidates)
    ShallowLists { avg_depth: f32 },
    /// Easy query (high agreement, low variance)
    Easy,
}

/// QPP-based fusion routing configuration.
#[derive(Debug, Clone)]
pub struct QppRoutingConfig {
    /// Difficulty threshold for switching strategies (default: 0.5)
    pub difficulty_threshold: f32,
    /// Fusion method for easy queries (default: CombSUM)
    pub easy_method: FusionMethod,
    /// Fusion method for hard queries (default: RRF with k=20)
    pub hard_method: FusionMethod,
    /// Minimum list depth to compute QPP (default: 5)
    pub min_depth: usize,
}

impl Default for QppRoutingConfig {
    fn default() -> Self {
        Self {
            difficulty_threshold: 0.5,
            easy_method: FusionMethod::combsum(),
            hard_method: FusionMethod::rrf(20),
            min_depth: 5,
        }
    }
}

/// Fuse with automatic QPP-based routing.
pub fn fuse_with_qpp<I, L>(
    lists: &[L],
    config: QppRoutingConfig,
) -> (Vec<(I, f32)>, QueryDifficulty)
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    // Implementation below
}
```

### Algorithm Details

```rust
/// Estimate query difficulty from result list characteristics.
fn estimate_difficulty<I, L>(lists: &[L], min_depth: usize) -> QueryDifficulty
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return QueryDifficulty {
            score: 1.0,
            reason: DifficultyReason::ShallowLists { avg_depth: 0.0 },
        };
    }

    // Compute overlap ratio (Jaccard similarity of top-K)
    let k = lists.iter()
        .map(|l| l.as_ref().len())
        .min()
        .unwrap_or(0)
        .min(min_depth);
    
    if k == 0 {
        return QueryDifficulty {
            score: 1.0,
            reason: DifficultyReason::ShallowLists { avg_depth: 0.0 },
        };
    }

    let mut all_ids: HashSet<I> = HashSet::new();
    let mut id_counts: HashMap<I, usize> = HashMap::new();
    
    for list in lists {
        for (id, _) in list.as_ref().iter().take(k) {
            all_ids.insert(id.clone());
            *id_counts.entry(id.clone()).or_insert(0) += 1;
        }
    }

    let overlap_count = id_counts.values().filter(|&&c| c > 1).count();
    let overlap_ratio = if all_ids.is_empty() {
        0.0
    } else {
        overlap_count as f32 / all_ids.len() as f32
    };

    // Compute coefficient of variation (CV) across lists
    let mut cvs = Vec::new();
    for list in lists {
        let scores: Vec<f32> = list.as_ref().iter().take(k).map(|(_, s)| *s).collect();
        if scores.is_empty() {
            continue;
        }
        let mean = scores.iter().sum::<f32>() / scores.len() as f32;
        let variance = scores.iter().map(|s| (s - mean).powi(2)).sum::<f32>() / scores.len() as f32;
        let std = variance.sqrt();
        let cv = if mean.abs() > 1e-9 { std / mean.abs() } else { 0.0 };
        cvs.push(cv);
    }
    let avg_cv = cvs.iter().sum::<f32>() / cvs.len() as f32;

    // Combine signals: low overlap + high variance = hard query
    let difficulty = {
        let overlap_signal = 1.0 - overlap_ratio; // Low overlap = harder
        let variance_signal = (avg_cv / 2.0).min(1.0); // High CV = harder
        (overlap_signal * 0.6 + variance_signal * 0.4).min(1.0)
    };

    let reason = if overlap_ratio < 0.3 {
        DifficultyReason::LowOverlap { overlap_ratio }
    } else if avg_cv > 1.0 {
        DifficultyReason::HighVariance { cv: avg_cv }
    } else if k < min_depth {
        DifficultyReason::ShallowLists { avg_depth: k as f32 }
    } else {
        DifficultyReason::Easy
    };

    QueryDifficulty { score: difficulty, reason }
}

pub fn fuse_with_qpp<I, L>(
    lists: &[L],
    config: QppRoutingConfig,
) -> (Vec<(I, f32)>, QueryDifficulty)
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    let difficulty = estimate_difficulty(lists, config.min_depth);
    
    let method = if difficulty.score >= config.difficulty_threshold {
        &config.hard_method
    } else {
        &config.easy_method
    };

    let results = method.fuse(lists)?;
    (results, difficulty)
}
```

### Integration Points

1. **Extend `FusionMethod`**:
```rust
pub enum FusionMethod {
    // ... existing ...
    QppRouted(QppRoutingConfig),
}
```

2. **Add to `FusionBuilder`**:
```rust
impl FusionBuilder {
    pub fn qpp_routed() -> Self {
        Self {
            method: FusionMethod::QppRouted(QppRoutingConfig::default()),
        }
    }
}
```

### Testing Strategy

```rust
#[test]
fn test_qpp_low_overlap() {
    // Lists with low overlap should be classified as hard
    let a = vec![("d1", 0.9), ("d2", 0.8)];
    let b = vec![("d3", 0.9), ("d4", 0.8)]; // No overlap
    let (result, difficulty) = fuse_with_qpp(&[&a, &b], QppRoutingConfig::default());
    
    assert!(difficulty.score > 0.5);
    assert!(matches!(difficulty.reason, DifficultyReason::LowOverlap { .. }));
}

#[test]
fn test_qpp_high_agreement() {
    // Lists with high overlap should be classified as easy
    let a = vec![("d1", 0.9), ("d2", 0.8), ("d3", 0.7)];
    let b = vec![("d1", 0.85), ("d2", 0.75), ("d3", 0.65)]; // High overlap
    let (result, difficulty) = fuse_with_qpp(&[&a, &b], QppRoutingConfig::default());
    
    assert!(difficulty.score < 0.5);
    assert!(matches!(difficulty.reason, DifficultyReason::Easy));
}
```

### Performance Considerations

- QPP computation: O(L×K) where K is min list depth
- Adds ~5-10% overhead, but improves quality for hard queries
- Can be cached per query if same lists are fused multiple times
- Overlap computation: O(K×L) for Jaccard similarity of top-K
- CV computation: O(K×L) for coefficient of variation

### Edge Cases

- **Shallow lists** (K < min_depth): Classified as hard, uses conservative strategy
- **All lists empty**: Returns empty result with difficulty=1.0
- **Perfect overlap** (overlap_ratio=1.0): Classified as easy, uses fast strategy
- **Zero variance**: CV=0, difficulty based only on overlap

---

## Plan 3: Additive Fusion Variant (ResFlow-style)

### Overview
Add explicit support for additive fusion of multi-task scores. Based on ResFlow (arXiv:2411.09705) which **empirically shows additive fusion outperforms multiplicative** for e-commerce ranking: `Score = α·CTR + β·CTCVR` consistently beats `CTR^α × CVR^β`. This is counterintuitive but validated in their online A/B tests (1.29% OPU increase).

**Key insight from ResFlow**: Multiplicative fusion is too sensitive to extreme values—if one score is very small, others lose voting rights. Additive fusion provides milder control and better aligns with online metrics.

### API Design

```rust
/// Additive multi-task fusion configuration (ResFlow-style).
#[derive(Debug, Clone)]
pub struct AdditiveMultiTaskConfig {
    /// Weight for first task (default: 0.5)
    pub weight_a: f32,
    /// Weight for second task (default: 0.5)
    pub weight_b: f32,
    /// Normalization method (default: ZScore for robustness)
    pub normalization: Normalization,
    /// Top-K truncation
    pub top_k: Option<usize>,
}

impl Default for AdditiveMultiTaskConfig {
    fn default() -> Self {
        Self {
            weight_a: 0.5,
            weight_b: 0.5,
            normalization: Normalization::ZScore, // ResFlow uses standardization
            top_k: None,
        }
    }
}

impl AdditiveMultiTaskConfig {
    pub fn new(weights: (f32, f32)) -> Self {
        Self {
            weight_a: weights.0,
            weight_b: weights.1,
            normalization: Normalization::ZScore,
            top_k: None,
        }
    }
}

/// Additive multi-task score fusion (ResFlow-style).
///
/// Formula: `score(d) = Σ (weight_i × normalized_score_i)`
///
/// ResFlow paper shows additive fusion outperforms multiplicative for e-commerce:
/// - Additive: `α·CTR + β·CTCVR` (better)
/// - Multiplicative: `CTR^α × CVR^β` (too sensitive to extreme values)
///
/// Use when fusing scores from multiple related tasks (CTR, CVR, relevance).
pub fn additive_multi_task<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
    weights: (f32, f32),
) -> Vec<(I, f32)> {
    additive_multi_task_with_config(
        results_a,
        results_b,
        AdditiveMultiTaskConfig::new(weights),
    )
}
```

### Algorithm Details

```rust
pub fn additive_multi_task_multi<I, L>(
    lists: &[(L, f32)], // (list, weight) pairs
    config: AdditiveMultiTaskConfig,
) -> Vec<(I, f32)>
where
    I: Clone + Eq + Hash,
    L: AsRef<[(I, f32)]>,
{
    if lists.is_empty() {
        return Vec::new();
    }

    // Normalize weights to sum to 1
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < WEIGHT_EPSILON {
        return Vec::new();
    }

    let estimated_size: usize = lists.iter().map(|(l, _)| l.as_ref().len()).sum();
    let mut scores: HashMap<I, f32> = HashMap::with_capacity(estimated_size);

    for (list, weight) in lists {
        let items = list.as_ref();
        if items.is_empty() {
            continue;
        }

        let normalized_weight = weight / total_weight;
        
        // Normalize scores based on config
        let normalized_items: Vec<(I, f32)> = match config.normalization {
            Normalization::ZScore => {
                let (mean, std) = zscore_params(items);
                items
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
            Normalization::MinMax => {
                let (norm, off) = min_max_params(items);
                items
                    .iter()
                    .map(|(id, s)| (id.clone(), (s - off) * norm))
                    .collect()
            }
            _ => items.to_vec(), // No normalization
        };

        // Add weighted normalized scores
        for (id, normalized_score) in normalized_items {
            let contribution = normalized_weight * normalized_score;
            *scores.entry(id).or_insert(0.0) += contribution;
        }
    }

    finalize(scores, config.top_k)
}
```

### Performance Considerations

- Same complexity as weighted fusion: O(L×N + U×log U)
- Normalization overhead: O(N) per list (same as existing methods)
- ResFlow reports no latency increase in production (110ms avg, 147ms p99)

### Edge Cases (from ResFlow paper)

- **Zero weights**: Returns empty result (same as weighted fusion)
- **All scores equal in a list**: Z-score normalization returns 0.0 for all (handled)
- **Extreme values**: Additive fusion is more robust than multiplicative (validated in ResFlow)

### Use Cases

- E-commerce ranking: CTR + CTCVR fusion (ResFlow's primary use case)
- Multi-task learning: combining scores from related prediction tasks
- Hybrid retrieval: BM25 + dense retriever scores (with standardization)

### Testing

```rust
#[test]
fn test_additive_multi_task() {
    let ctr = vec![("d1", 0.9), ("d2", 0.8)];
    let ctcvr = vec![("d1", 0.7), ("d3", 0.9)]; // d3 only in CTCVR
    
    // ResFlow's optimal: CTR + CTCVR*20 (from their experiments)
    let result = additive_multi_task(&ctr, &ctcvr, (1.0, 20.0));
    
    // d1 appears in both, should rank highest
    assert_eq!(result[0].0, "d1");
}

#[test]
fn test_additive_vs_multiplicative() {
    // Test that additive is more robust to extreme values
    let a = vec![("d1", 0.01), ("d2", 0.9)]; // d1 has very low score
    let b = vec![("d1", 0.9), ("d2", 0.9)];
    
    let additive = additive_multi_task(&a, &b, (0.5, 0.5));
    // d1 should still rank reasonably in additive (not zeroed out)
    // In multiplicative, d1 would be near-zero due to 0.01
}
```

---

## Implementation Order

1. **Standardized fusion** (Plan 1) - Highest impact, reuses existing `zscore_params`, validated in ERANK
2. **Additive multi-task fusion** (Plan 3) - High impact for e-commerce, validated in ResFlow online tests
3. **QPP routing** (Plan 2) - Medium complexity, adds intelligence but requires more validation

## Validation Strategy

For each plan:
1. Unit tests for edge cases (empty lists, zero variance, etc.)
2. Property tests (invariants: monotonicity, idempotency)
3. Benchmark against existing methods on real datasets
4. Integration tests with rank-refine pipeline

