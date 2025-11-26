# rank-fusion

Rank fusion algorithms for combining retrieval results.

[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

## Usage

```toml
[dependencies]
rank-fusion = "0.1"
```

```rust
use rank_fusion::{rrf, RrfConfig};

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2), ("doc3", 9.8)];
let dense = vec![("doc2", 0.92), ("doc1", 0.87), ("doc4", 0.85)];

let fused = rrf(bm25, dense, RrfConfig::default());
// doc1, doc2 rank highest (appear in both)
```

## Algorithms

| Function | Method | Uses Scores |
|----------|--------|-------------|
| `rrf` | Reciprocal Rank Fusion | No |
| `combsum` | Sum of normalized scores | Yes |
| `combmnz` | Sum × overlap count | Yes |
| `borda` | Borda count | No |
| `weighted` | Weighted combination | Yes |

**RRF** ignores scores, uses only rank position. Robust when combining systems with different score distributions (BM25 vs cosine similarity).

**CombMNZ** rewards documents appearing in multiple lists.

**Weighted** for when you trust one retriever more.

## Configuration

```rust
// RRF: k controls how much top ranks dominate
let config = RrfConfig::new(60); // default

// Weighted: control blend
let config = WeightedConfig::new(0.3, 0.7); // 30% list A, 70% list B
```

## Multiple Lists

```rust
use rank_fusion::rrf_multi;

let lists = vec![bm25, dense, sparse];
let fused = rrf_multi(&lists, RrfConfig::default());
```

## Performance

Zero dependencies. Runs in microseconds for typical workloads (100-1000 results per list).

## License

MIT OR Apache-2.0
