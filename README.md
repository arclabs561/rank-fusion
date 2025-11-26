# rank-fusion

Fast rank fusion for hybrid search.

[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE-MIT)

Zero-dependency implementations of rank fusion algorithms for combining results from multiple retrieval systems (BM25 + vectors, hybrid search, RAG).

## Algorithms

| Function | Description |
|----------|-------------|
| `rrf` | Reciprocal Rank Fusion — robust, parameter-free |
| `combsum` | Sum of normalized scores |
| `combmnz` | Sum × overlap count (rewards docs in multiple lists) |
| `borda` | Borda count voting |
| `weighted` | Configurable weights with optional normalization |

## Usage

```rust
use rank_fusion::{rrf, RrfConfig};

let sparse = vec![("doc1", 0.9), ("doc2", 0.7)];  // BM25
let dense = vec![("doc2", 0.8), ("doc3", 0.6)];   // vectors

let fused = rrf(sparse, dense, RrfConfig::default());
// doc2 ranks first (appears in both)
```

## Performance

~10M elements/sec on typical workloads. Use `rrf_into` for zero-allocation hot paths.

## License

MIT OR Apache-2.0
