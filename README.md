# rank-fusion

Combine ranked lists from multiple retrievers. Provides RRF, CombMNZ, Borda, DBSF, RBC, Condorcet, and 10+ fusion algorithms with full explainability, hyperparameter optimization, and IR metrics. Zero dependencies.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

```
cargo add rank-fusion
```

## Why Rank Fusion?

Hybrid search combines multiple retrievers (BM25, dense embeddings, sparse vectors) to get the best of each. This requires merging their results.

**Problem**: Different retrievers use incompatible score scales. BM25 might score 0-100, while dense embeddings score 0-1. Normalization is fragile and requires tuning.

**RRF (Reciprocal Rank Fusion)**: Ignores scores and uses only rank positions. The formula `1/(k + rank)` ensures:
- Top positions dominate (rank 0 gets 1/60 = 0.017, rank 5 gets 1/65 = 0.015)
- Multiple list agreement is rewarded (documents appearing in both lists score higher)
- No normalization needed (works with any score distribution)

**Example**: Document "d2" appears at rank 0 in BM25 list and rank 1 in dense list:
- RRF score = 1/(60+0) + 1/(60+1) = 0.0167 + 0.0164 = 0.0331
- This beats "d1" (only in BM25 at rank 0: 0.0167) and "d3" (only in dense at rank 1: 0.0164)

RRF finds consensus across retrievers. No normalization needed, works with any score distribution.

## What This Is

Fusion algorithms for hybrid search:

| Scenario | Algorithm |
|----------|-----------|
| BM25 + dense embeddings | `rrf` (rank-based) |
| Variable-length lists | `rbc` (Rank-Biased Centroids) |
| Multiple retrievers, different scales | `rrf_multi` |
| Same-scale scores | `combsum`, `combmnz` |
| Trust one retriever more | `weighted`, `rrf_weighted` |
| Different distributions | `dbsf` (z-score) |
| Robust to outliers | `condorcet`, `combmed` |
| Baselines | `combmax`, `combanz` |

**What this is NOT**: embedding generation, vector search, or scoring embeddings. See [rank-refine](https://crates.io/crates/rank-refine) for scoring embeddings.

## Usage

```rust
use rank_fusion::rrf;

let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
let dense = vec![("d2", 0.9), ("d3", 0.8)];

let fused = rrf(&bm25, &dense);
// [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
```

### Realistic Example

```rust
use rank_fusion::rrf;

// BM25 results (50 items, scores 0-100)
let bm25_results = vec![
    ("doc_123", 87.5),
    ("doc_456", 82.3),
    ("doc_789", 78.1),
    // ... 47 more results
];

// Dense embedding results (50 items, cosine similarity 0-1)
let dense_results = vec![
    ("doc_456", 0.92),
    ("doc_123", 0.88),
    ("doc_999", 0.85),
    // ... 47 more results
];

// RRF finds consensus: doc_456 appears high in both lists
let fused = rrf(&bm25_results, &dense_results);
// doc_456 wins (rank 1 in BM25, rank 0 in dense)
// doc_123 second (rank 0 in BM25, rank 1 in dense)
// doc_789 third (rank 2 in BM25, not in dense top-50)
```

## API

### Rank-based (ignores scores)

| Function | Formula | Use |
|----------|---------|-----|
| `rrf(a, b)` | 1/(k + rank) | Different scales |
| `isr(a, b)` | 1/√(k + rank) | Lower ranks matter more |
| `borda(a, b)` | N - rank | Simple voting |
| `rbc(a, b)` | (1-p)^rank / (1-p^N) | Variable-length lists |
| `condorcet(a, b)` | Pairwise voting | Robust to outliers |

### Score-based

| Function | Formula | Use |
|----------|---------|-----|
| `combsum(a, b)` | Σ scores | Same scale |
| `combmnz(a, b)` | sum × count | Reward overlap |
| `dbsf(a, b)` | z-score | Different distributions |
| `weighted(a, b, config)` | weighted sum | Custom weights |
| `combmax(a, b)` | max(scores) | Baseline, favor high scores |
| `combmed(a, b)` | median(scores) | Robust to outliers |
| `combanz(a, b)` | mean(scores) | Average instead of sum |

### Multi-list

All functions have `*_multi` variants:

```rust
use rank_fusion::{rrf_multi, RrfConfig};

let lists = vec![&bm25[..], &dense[..], &sparse[..]];
let fused = rrf_multi(&lists, RrfConfig::default());
```

### Weighted RRF

```rust
use rank_fusion::rrf_weighted;

let weights = [1.0, 2.0, 0.5];  // per-retriever
let fused = rrf_weighted(&lists, &weights, config)?;
```

### Explainability

Debug and analyze fusion results with full provenance:

```rust
use rank_fusion::explain::{rrf_explain, analyze_consensus, attribute_top_k, RetrieverId};
use rank_fusion::RrfConfig;

let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
let dense = vec![("d2", 0.9), ("d3", 0.8)];

let retrievers = vec![
    RetrieverId::new("bm25"),
    RetrieverId::new("dense"),
];

// Get results with full provenance
let explained = rrf_explain(
    &[&bm25[..], &dense[..]],
    &retrievers,
    RrfConfig::default(),
);

// Each result shows which retrievers contributed and how
for result in &explained {
    println!("{}: score={:.6}, consensus={:.1}%",
        result.id, result.score,
        result.explanation.consensus_score * 100.0);
    for source in &result.explanation.sources {
        println!("  {}: rank {}, contribution {:.6}",
            source.retriever_id,
            source.original_rank.unwrap_or(999),
            source.contribution);
    }
}

// Analyze consensus patterns
let consensus = analyze_consensus(&explained);
println!("High consensus: {:?}", consensus.high_consensus);
println!("Single source: {:?}", consensus.single_source);

// Attribute top-k to retrievers
let attribution = attribute_top_k(&explained, 5);
for (retriever, stats) in &attribution {
    println!("{}: {} docs in top-5, {} unique",
        retriever, stats.top_k_count, stats.unique_docs);
}
```

See [`examples/explainability.rs`](examples/explainability.rs) for a complete example.

## Formulas

### Notation

- $d$: Document identifier
- $R$: Set of all retrievers
- $r$: A single retriever (element of $R$)
- $R_d$: Set of retrievers containing document $d$
- $\text{rank}_r(d)$: 0-indexed rank of document $d$ in retriever $r$ (top result = 0)
- $s_r(d)$: Score of document $d$ from retriever $r$
- $N$: Total number of documents in a list
- $k$: Smoothing constant (default 60 for RRF)

### RRF (Reciprocal Rank Fusion)

**RRF (Reciprocal Rank Fusion)**: Ignores score magnitudes and uses only rank positions. Formula:

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where $R$ is the set of retrievers, $k$ is a smoothing constant (default 60), and $\text{rank}_r(d)$ is the 0-indexed rank of document $d$ in retriever $r$ (top result = 0). From Cormack et al. (2009).

### Why k=60?

The k parameter controls how sharply top positions dominate. Cormack et al. (2009) tested k values from 1 to 100 and found k=60 balances:
- Top position emphasis (rank 0 vs rank 5: 1.1x ratio)
- Consensus across lists (lower k overweights single-list agreement)
- Robustness across datasets

**Sensitivity analysis**:

| k | rank 0 | rank 5 | rank 10 | Ratio (0 vs 5) | Use Case |
|---|--------|--------|---------|----------------|----------|
| 10 | 0.100 | 0.067 | 0.050 | 1.5x | Top positions highly reliable |
| 60 | 0.017 | 0.015 | 0.014 | 1.1x | Default for most scenarios |
| 100 | 0.010 | 0.0095 | 0.0091 | 1.05x | Want uniform contribution |

**When to tune**:
- k=20-40: When top retrievers are highly reliable, want strong consensus
- k=60: Default for most hybrid search scenarios
- k=100+: When lower-ranked items are still valuable, want broad agreement

**Visual example**:

```
BM25 list:        Dense list:
rank 0: d1 (12.5)  rank 0: d2 (0.9)
rank 1: d2 (11.0)  rank 1: d3 (0.8)
rank 2: d3 (10.5)  rank 2: d1 (0.7)

RRF scores (k=60):
d1: 1/(60+0) + 1/(60+2) = 0.0167 + 0.0161 = 0.0328
d2: 1/(60+1) + 1/(60+0) = 0.0164 + 0.0167 = 0.0331 (wins)
d3: 1/(60+2) + 1/(60+1) = 0.0161 + 0.0164 = 0.0325

Final ranking: [d2, d1, d3]
```

**CombMNZ**: Rewards documents appearing in multiple lists (consensus). Multiplies the sum of scores by the number of lists containing the document:

$$\text{score}(d) = \text{count}(d) \times \sum_r s_r(d)$$

**Example**: Document "d1" appears in 2 lists with scores [0.8, 0.7], while "d2" appears in 1 list with score 0.9:
- CombSUM: d1 = 0.8 + 0.7 = 1.5, d2 = 0.9 (d1 wins)
- CombMNZ: d1 = 2 × 1.5 = 3.0, d2 = 1 × 0.9 = 0.9 (d1 wins by larger margin)

**Borda Count**: Each position gets points equal to how many documents it beats. Formula:

$$\text{Borda}(d) = \sum_{r \in R} (N - \text{rank}_r(d))$$

where $N$ is the total number of documents in list $r$, and $\text{rank}_r(d)$ is 0-indexed.

**Example**: Two lists, each with 3 documents:
```
List 1: [d1, d2, d3]  (N=3)
List 2: [d2, d1, d3]  (N=3)
```

Borda scores:
- d1: (3-0) + (3-1) = 3 + 2 = 5
- d2: (3-1) + (3-0) = 2 + 3 = 5 (tie)
- d3: (3-2) + (3-2) = 1 + 1 = 2

Both d1 and d2 win, reflecting that they appear high in both lists.

**DBSF (Distribution-Based Score Fusion)**: Normalizes scores using z-scores to handle different distributions:

$$s' = \text{clip}\left(\frac{s - \mu}{\sigma}, -3, 3\right)$$

where $\mu$ and $\sigma$ are the mean and standard deviation of scores from that retriever. Clipping to $[-3, 3]$ bounds outliers: 99.7% of values in a normal distribution fall within ±3σ. This prevents one extreme score from dominating.

**Example**: Retriever A has scores [10, 12, 15, 18, 20] (mean=15, σ=4). Document with score 25 gets z-score (25-15)/4 = 2.5. Document with score 30 gets clipped to 3.0 (would be 3.75 without clipping).

### Relationship Between Algorithms

**CombMNZ vs CombSUM**: CombMNZ is CombSUM multiplied by consensus count:

$$\text{CombMNZ}(d) = \text{count}(d) \times \text{CombSUM}(d)$$

**ISR vs RRF**: ISR uses square root instead of linear reciprocal:

$$\text{ISR}(d) = \sum_{r \in R} \frac{1}{\sqrt{k + \text{rank}_r(d)}}$$

This gives lower-ranked items more weight. Use ISR when you want to consider items beyond the top 10-20.

**DBSF vs CombSUM**: DBSF is CombSUM with z-score normalization instead of min-max:

$$\text{DBSF}(d) = |R_d| \cdot \sum_{r \in R_d} \text{clip}\left(\frac{s_r(d) - \mu_r}{\sigma_r}, -3, 3\right)$$

Use DBSF when score distributions differ significantly between retrievers.

## Benchmarks

Measured on Apple M3 Max with `cargo bench`:

| Operation | Items | Time |
|-----------|-------|------|
| `rrf` | 100 | 13μs |
| `rrf` | 1000 | 159μs |
| `combsum` | 100 | 14μs |
| `combmnz` | 100 | 13μs |
| `borda` | 100 | 13μs |
| `rrf_multi` (5 lists) | 100 | 38μs |

These timings are suitable for real-time fusion of 100-1000 item lists.

## Vendoring

The code can be vendored if you prefer not to add a dependency:

- `src/lib.rs` is self-contained (~2000 lines)
- Zero dependencies
- All algorithms in one file

## Choosing a Fusion Method

Start here: Do your retrievers use compatible score scales?

```
├─ No (BM25: 0-100, dense: 0-1) → Use rank-based
│  ├─ Need strong consensus? → RRF (k=60)
│  └─ Lower ranks still valuable? → ISR (k=1)
│
└─ Yes (both 0-1, both cosine similarity) → Use score-based
   ├─ Want to reward overlap? → CombMNZ
   ├─ Simple sum? → CombSUM
   ├─ Different distributions? → DBSF (z-score normalization)
   └─ Trust one retriever more? → Weighted
```

**When RRF underperforms**:

RRF is typically 3-4% lower NDCG than CombSUM when score scales are compatible (OpenSearch BEIR benchmarks). Trade-off:
- RRF: Robust to scale mismatches, no tuning needed
- CombSUM: Better quality when scales match, requires normalization

**Use RRF when**:
- Score scales are unknown or incompatible
- You want zero-configuration fusion
- Robustness is more important than optimal quality

**Use CombSUM when**:
- Scores are on the same scale (both cosine, both BM25, etc.)
- You can normalize reliably
- Quality is more important than convenience

See [DESIGN.md](DESIGN.md) for algorithm details.

## Explainability

The `explain` module provides variants of fusion functions that return full provenance information, showing:

- **Which retrievers** contributed each document
- **Original ranks and scores** from each retriever
- **Contribution amounts** showing how much each source added to the final score
- **Consensus scores** indicating how many retrievers agreed on each document

This is critical for debugging RAG pipelines: you can see if your expensive cross-encoder is actually helping, identify retriever disagreement patterns, and understand why certain documents ranked where they did.

**Use cases:**
- Debugging retrieval failures ("Why did this relevant doc rank so low?")
- A/B testing retrievers ("Is my new embedding model actually improving results?")
- Building user trust ("This answer came 60% from our docs, 40% from community forum")
- Identifying index staleness or embedding drift

See [`examples/explainability.rs`](examples/explainability.rs) for a complete example.

## Normalization

Score normalization is now a first-class concern. Use `normalize_scores()` to explicitly control how scores are normalized before fusion:

```rust
use rank_fusion::{normalize_scores, Normalization};

let scores = vec![("d1", 10.0), ("d2", 5.0), ("d3", 0.0)];

// Min-max normalization (default for CombSUM/CombMNZ)
let normalized = normalize_scores(&scores, Normalization::MinMax);

// Z-score normalization (used by DBSF)
let z_normalized = normalize_scores(&scores, Normalization::ZScore);

// Sum normalization (preserves relative magnitudes)
let sum_normalized = normalize_scores(&scores, Normalization::Sum);

// Rank-based (ignores score magnitudes)
let rank_normalized = normalize_scores(&scores, Normalization::Rank);
```

## Runtime Strategy Selection

Use the `FusionStrategy` enum for dynamic method selection:

```rust
use rank_fusion::strategy::FusionStrategy;

// Select method at runtime
let method = if use_scores {
    FusionStrategy::combsum()
} else {
    FusionStrategy::rrf(60)
};

let result = method.fuse(&[&list1[..], &list2[..]]);
println!("Using method: {}", method.name());
```

## Hyperparameter Optimization

Optimize fusion parameters using ground truth (qrels):

```rust
use rank_fusion::optimize::{optimize_fusion, OptimizeConfig, OptimizeMetric, ParamGrid};
use rank_fusion::FusionMethod;

// Relevance judgments
let qrels = std::collections::HashMap::from([
    ("doc1", 2), // highly relevant
    ("doc2", 1), // relevant
]);

// Retrieval runs
let runs = vec![
    vec![("doc1", 0.9), ("doc2", 0.8)],
    vec![("doc2", 0.9), ("doc1", 0.7)],
];

// Optimize RRF k parameter
let config = OptimizeConfig {
    method: FusionMethod::Rrf { k: 60 },
    metric: OptimizeMetric::Ndcg { k: 10 },
    param_grid: ParamGrid::RrfK {
        values: vec![20, 40, 60, 100],
    },
};

let optimized = optimize_fusion(&qrels, &runs, config);
println!("Best k: {}, NDCG@10: {:.4}", optimized.best_params, optimized.best_score);
```

## Evaluation Metrics

The crate includes standard IR metrics for evaluation:

```rust
use rank_fusion::{ndcg_at_k, mrr, recall_at_k};

let results = vec![("d1", 1.0), ("d2", 0.9), ("d3", 0.8)];
let qrels = std::collections::HashMap::from([
    ("d1", 2), // highly relevant
    ("d2", 1), // relevant
]);

let ndcg = ndcg_at_k(&results, &qrels, 10);
let reciprocal_rank = mrr(&results, &qrels);
let recall = recall_at_k(&results, &qrels, 10);
```

## See Also

- [rank-refine](https://crates.io/crates/rank-refine): score with embeddings (cosine, MaxSim)
- [DESIGN.md](DESIGN.md): algorithm details and edge cases

## License

MIT OR Apache-2.0
