# rank-fusion

**Combine ranked lists from multiple search systems.** Zero dependencies.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

## Why This Library?

Hybrid search combines multiple retrievers:

```
Query → BM25 (lexical) → ranked list A
      → Dense (semantic) → ranked list B
      → Sparse (SPLADE) → ranked list C
                           ↓
                    Fusion → final ranking
```

**This crate is the fusion step.** It merges ranked lists into one.

| What | This Crate | Other Tools |
|------|------------|-------------|
| **Input** | Ranked lists with scores | Embeddings, raw text |
| **Output** | Combined ranking | Model inference |
| **Dependencies** | 0 | Many |

### vs rank-refine

| | rank-fusion | [rank-refine](https://crates.io/crates/rank-refine) |
|-|-------------|-------------|
| **Input** | Ranked lists `(id, score)` | Embeddings `Vec<f32>` |
| **Use case** | Merge BM25 + dense + sparse | Rescore with similarity |
| **Example** | RRF, CombSUM | cosine, MaxSim, MMR |

Use **rank-fusion** when combining retriever outputs (no embeddings needed).
Use **rank-refine** when you have embeddings and want semantic reranking.

## Quick Start

```rust
use rank_fusion::rrf;

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2)];
let dense = vec![("doc2", 0.92), ("doc1", 0.80)];

let fused = rrf(&bm25, &dense);
// doc2 ranked high in both → comes first
```

## The Problem

Different retrievers, different scales:

```
BM25:   doc1: 15.2,  doc2: 12.1   (unbounded, log-based)
Dense:  doc2: 0.92,  doc1: 0.80   (0 to 1, cosine)
Sparse: doc1: 8.3,   doc3: 7.1    (unbounded, sum of weights)
```

Naive score addition fails — BM25 dominates due to larger values.

## Algorithms

### RRF (Reciprocal Rank Fusion) — Default

Ignores scores. Only position matters:

$$\text{RRF}(d) = \sum_{r} \frac{1}{k + \text{rank}_r(d)}$$

```rust
use rank_fusion::{rrf, rrf_with_config, RrfConfig};

let fused = rrf(&a, &b);                              // k=60 default
let fused = rrf_with_config(&a, &b, RrfConfig::new(20));  // k=20: top-heavy
```

**Properties**: Outlier-resistant, no tuning, works across any scale.

### Weighted RRF

Per-retriever weights for heterogeneous systems:

```rust
use rank_fusion::rrf_weighted;

let lists = vec![bm25, dense, sparse];
let weights = vec![1.0, 2.0, 0.5];  // favor dense
let fused = rrf_weighted(&lists, &weights, config)?;
```

### Score-Based Methods

| Function | Formula | When to Use |
|----------|---------|-------------|
| `combsum` | Σ normalized scores | Same-scale, trust magnitude |
| `combmnz` | combsum × overlap count | Reward agreement |
| `dbsf` | z-score + clipping | Different distributions |
| `weighted` | Σ weight × score | Custom retriever importance |

```rust
use rank_fusion::{combsum, combmnz, dbsf, weighted};

let fused = combsum(&a, &b);      // simple sum
let fused = combmnz(&a, &b);      // reward overlap
let fused = dbsf(&a, &b);         // z-score normalization
let fused = weighted(&a, &b, 0.7, 0.3);  // 70% a, 30% b
```

### Rank-Based Methods

| Function | Formula | When to Use |
|----------|---------|-------------|
| `rrf` | 1/(k + rank) | Different scales (default) |
| `isr` | 1/√(k + rank) | Lower ranks matter more |
| `borda` | N - rank | Simple voting |

```rust
use rank_fusion::{isr, borda};

let fused = isr(&a, &b);    // inverse square root
let fused = borda(&a, &b);  // Borda count
```

### Multiple Lists

All functions have `*_multi` variants:

```rust
use rank_fusion::{rrf_multi, combsum_multi, RrfConfig, FusionConfig};

let lists = vec![bm25, dense, sparse, reranker];
let fused = rrf_multi(&lists, RrfConfig::default());
let fused = combsum_multi(&lists, FusionConfig::default());
```

## Configuration

```rust
use rank_fusion::{RrfConfig, FusionConfig};

// RRF tuning
let config = RrfConfig::default()
    .with_k(60)      // smoothing constant
    .with_top_k(10); // return top 10 only

// Score-based tuning
let config = FusionConfig::default()
    .with_top_k(10);
```

## Performance Notes

OpenSearch benchmarks (BEIR datasets) show:
- RRF: ~3-4% lower NDCG than score-based methods
- RRF: ~1-2% faster latency
- RRF excels when score scales are incompatible

See [DESIGN.md](DESIGN.md) for mathematical foundations and algorithm trade-offs.

## License

MIT OR Apache-2.0
