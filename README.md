# rank-fusion

Combine ranked lists from multiple retrievers. Provides RRF (Reciprocal Rank Fusion), CombMNZ, Borda, DBSF, and weighted fusion. Zero dependencies.

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
| Multiple retrievers, different scales | `rrf_multi` |
| Same-scale scores | `combsum`, `combmnz` |
| Trust one retriever more | `weighted`, `rrf_weighted` |
| Different distributions | `dbsf` (z-score) |

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

### Score-based

| Function | Formula | Use |
|----------|---------|-----|
| `combsum(a, b)` | Σ scores | Same scale |
| `combmnz(a, b)` | sum × count | Reward overlap |
| `dbsf(a, b)` | z-score | Different distributions |
| `weighted(a, b, config)` | weighted sum | Custom weights |

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

## Formulas

**RRF (Reciprocal Rank Fusion)**: Ignores score magnitudes and uses only rank positions. Formula:

$$\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$$

Default k=60. Rank is 0-indexed. From Cormack et al. (2009).

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

**CombMNZ**: Multiplies sum by count of lists containing the document:
$$\text{score}(d) = \text{count}(d) \times \sum_r s_r(d)$$

**DBSF (Distribution-Based Score Fusion)**: Z-score normalization:
$$s' = \frac{s - \mu}{\sigma}, \quad \text{clipped to } [-3, 3]$$

Clipping to ±3σ bounds outliers. About 99.7% of normal distribution is within 3σ.

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

## See Also

- [rank-refine](https://crates.io/crates/rank-refine): score with embeddings (cosine, MaxSim)
- [DESIGN.md](DESIGN.md): algorithm details and edge cases

## License

MIT OR Apache-2.0
