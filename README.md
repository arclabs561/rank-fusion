# rank-fusion

Combine ranked lists from multiple search systems.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

**Zero dependencies.** Adds ~0.1s to compile, ~10KB to binary.

## When to Use

You have multiple retrievers (BM25, dense, sparse) and need one combined ranking.

```rust
use rank_fusion::rrf;

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2)];
let dense = vec![("doc2", 0.92), ("doc1", 0.80)];

let fused = rrf(&bm25, &dense);
// doc2 ranked high in both → comes first
```

## When NOT to Use

- Single retriever → no fusion needed
- Vector DB with built-in RRF → use theirs (qdrant, weaviate, etc.)
- Need reranking with embeddings → use [`rank-refine`](https://crates.io/crates/rank-refine)

## The Problem

Different systems, different score scales:

```
BM25:   doc1: 15.2,  doc2: 12.1   (unbounded)
Dense:  doc2: 0.92,  doc1: 0.80   (0 to 1)
```

Adding scores directly fails — BM25 dominates.

## RRF: The Default Choice

Reciprocal Rank Fusion ignores scores. Only position matters:

```math
\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}
```

- Outlier-resistant
- No tuning needed
- Works across any score scale

## Algorithms

| Function | Uses Scores | Best For |
|----------|-------------|----------|
| `rrf` | No | Different scales (default) |
| `isr` | No | When lower ranks matter more |
| `combsum` | Yes | Same-scale scores |
| `combmnz` | Yes | Reward overlap |
| `borda` | No | Simple voting |
| `weighted` | Yes | Prefer one retriever |
| `dbsf` | Yes | Z-score normalization |

All have `*_multi` variants for 3+ lists.

## Configuration

```rust
use rank_fusion::{rrf, rrf_with_config, RrfConfig};

let a = vec![("doc1", 1.0)];
let b = vec![("doc2", 1.0)];

// k=60 is default
let fused = rrf(&a, &b);

// k=20: top results dominate
let fused = rrf_with_config(&a, &b, RrfConfig::new(20));

// k=100: more uniform contribution
let fused = rrf_with_config(&a, &b, RrfConfig::new(100));
```

## Multiple Lists

```rust
use rank_fusion::{rrf_multi, RrfConfig};

let bm25 = vec![("d1", 1.0)];
let dense = vec![("d2", 1.0)];
let sparse = vec![("d1", 1.0)];

let fused = rrf_multi(&[bm25, dense, sparse], RrfConfig::default());
```

## Design

See [DESIGN.md](DESIGN.md) for:
- Mathematical foundations (social choice theory)
- Why RRF over Kemeny optimal
- Trade-offs between algorithms

## License

MIT OR Apache-2.0
