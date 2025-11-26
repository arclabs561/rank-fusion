# rerank

Fast rank fusion algorithms for hybrid search systems.

[![Crates.io](https://img.shields.io/crates/v/rerank.svg)](https://crates.io/crates/rerank)
[![Documentation](https://docs.rs/rerank/badge.svg)](https://docs.rs/rerank)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

## Overview

`rerank` provides zero-dependency implementations of rank fusion algorithms used to combine results from multiple retrieval systems. This is essential for hybrid search in RAG (Retrieval Augmented Generation) applications.

## Features

| Method | Type | Best For |
|--------|------|----------|
| **RRF** | Rank-based | General use, parameter-free |
| **CombSUM** | Score-based | When scores are comparable |
| **CombMNZ** | Score-based | Rewarding overlap |
| **Borda** | Rank-based | Voting scenarios |
| **Weighted** | Score-based | Tuned weights |

## Performance

Benchmarked on M1 Mac (100 results per list):

| Method | Latency | Throughput |
|--------|---------|------------|
| RRF | 13 µs | 7.8 Melem/s |
| RRF (preallocated) | 9.6 µs | 10.4 Melem/s |
| CombMNZ | 13.5 µs | 7.4 Melem/s |

## Usage

```rust
use rerank::{fuse_rrf, fuse_combmnz, RrfConfig};

// Results from BM25 (sparse) retrieval
let sparse = vec![
    ("doc1".to_string(), 0.9),
    ("doc2".to_string(), 0.7),
];

// Results from dense (vector) retrieval
let dense = vec![
    ("doc2".to_string(), 0.85),
    ("doc3".to_string(), 0.6),
];

// RRF fusion (recommended default)
let fused = fuse_rrf(sparse.clone(), dense.clone(), RrfConfig::default());
assert_eq!(fused[0].0, "doc2"); // appears in both

// CombMNZ for overlap reward
let fused = fuse_combmnz(&sparse, &dense);
```

### Zero-Allocation Path

For hot paths, use preallocated output:

```rust
use rerank::{fuse_rrf_into, RrfConfig};

let mut output = Vec::with_capacity(200);
fuse_rrf_into(&sparse, &dense, RrfConfig::default(), &mut output);
```

### 3+ Sources

```rust
use rerank::{fuse_rrf_multi, RrfConfig};

let lists = vec![sparse, dense, knowledge_graph_results];
let fused = fuse_rrf_multi(&lists, RrfConfig::default());
```

## When to Use What

- **RRF (k=60)**: Start here. Works well without tuning.
- **CombMNZ**: When overlap between retrievers indicates relevance.
- **Weighted**: When you've tuned weights on your dataset.
- **Borda**: Voting-style aggregation.

## no_std Support

```toml
[dependencies]
rerank = { version = "0.1", default-features = false, features = ["alloc"] }
```

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.

