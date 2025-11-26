# rank-fusion

Rank fusion for hybrid search.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)
[![MSRV](https://img.shields.io/badge/MSRV-1.74-blue)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

## The Score Incompatibility Problem

Hybrid search combines multiple retrievers. The problem: their scores are incompatible.

```
BM25:   "rust programming"  →  doc1: 15.2,  doc2: 12.1,  doc3: 8.5
Dense:  "rust programming"  →  doc2: 0.92,  doc4: 0.85,  doc1: 0.80
```

BM25 scores are unbounded (0 to 20+). Cosine similarity is bounded (-1 to 1).
You can't add 15.2 + 0.92 and get anything meaningful.

**Naive approaches fail:**
- Raw addition: BM25 dominates (larger numbers)
- Min-max normalization: Outliers distort the scale
- Z-score normalization: Assumes normal distribution (often wrong)

## Why RRF Works

Reciprocal Rank Fusion (Cormack et al., 2009) sidesteps scores entirely:

```
RRF(d) = Σ 1/(k + rank(d))
```

A document at rank 1 in both lists gets `1/(60+1) + 1/(60+1) = 0.033`.
A document at rank 1 in one list and absent from another gets `1/(60+1) = 0.016`.

RRF's advantages:
- **Outlier resistant**: Scores don't matter, only ranks
- **No scale assumptions**: Works across any retriever
- **Rewards consensus**: Documents ranked highly in multiple lists rise to the top

## Quick Start

```rust
use rank_fusion::prelude::*;

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2), ("doc3", 8.0)];
let dense = vec![("doc2", 0.92), ("doc4", 0.85), ("doc1", 0.80)];

// RRF ignores scores, uses ranks only
let fused = rrf(bm25, dense, RrfConfig::default());
// doc2 appears in both → ranked highest
```

## Algorithms

| Function | Uses Scores | Best For |
|----------|-------------|----------|
| `rrf` | No | Incompatible score scales (default choice) |
| `combsum` | Yes | Same-scale scores you trust |
| `combmnz` | Yes | Reward multi-retriever agreement |
| `borda` | No | Simple voting baseline |
| `weighted` | Yes | When one retriever is clearly better |

All have `*_multi` variants for 3+ lists.

## Choosing an Algorithm

**Start with RRF.** It's the safest default because it makes no assumptions about score distributions.

Use score-based methods (`combsum`, `weighted`) only when:
1. Your retrievers produce same-scale scores (e.g., two cosine similarity systems)
2. You've validated that score magnitudes correlate with relevance
3. You need fine-grained control over retriever contributions

## RRF k Parameter

The `k` parameter (default: 60) controls how quickly scores decay with rank:

| k | Effect | Use When |
|---|--------|----------|
| 10-30 | Aggressive — top results dominate | High confidence in top-k |
| 50-70 | Balanced | General hybrid search |
| 100+ | Conservative — rewards consensus | Noisy or experimental retrievers |

```rust
// Aggressive (trust top results)
let config = RrfConfig::new(20);

// Conservative (reward agreement)
let config = RrfConfig::new(100);
```

## Weighted RRF

When retrievers have known quality differences:

```rust
use rank_fusion::rrf_weighted;

let weights = [0.3, 0.7];  // Trust dense 2x more
let fused = rrf_weighted(&[&bm25, &dense], &weights, RrfConfig::default())?;
```

## Multi-List Fusion

For 3+ retrievers (keyword, dense, sparse, ColBERT, etc.):

```rust
use rank_fusion::rrf_multi;

let lists = vec![bm25_results, dense_results, sparse_results];
let fused = rrf_multi(&lists, RrfConfig::default());
```

## FusionMethod (Unified API)

For runtime algorithm selection:

```rust
use rank_fusion::FusionMethod;

let method = FusionMethod::Rrf { k: 60 };
let fused = method.fuse(&sparse, &dense);

// Or weighted
let method = FusionMethod::Weighted { w1: 0.3, w2: 0.7 };
let fused = method.fuse(&sparse, &dense);
```

## Related

- [`rank-refine`](https://crates.io/crates/rank-refine) — Reranking algorithms (ColBERT, MRL, cross-encoder)

## References

- Cormack, Clarke, Buettcher (2009). "Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods"
- [OpenSearch: Introducing RRF for Hybrid Search](https://opensearch.org/blog/introducing-reciprocal-rank-fusion-hybrid-search/)

## License

MIT OR Apache-2.0
