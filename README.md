# rank-fusion

Rank fusion for hybrid search — combine results from multiple retrievers.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)
[![MSRV](https://img.shields.io/badge/MSRV-1.74-blue)](https://blog.rust-lang.org/2023/11/16/Rust-1.74.0.html)

## Why This Library?

Hybrid search (BM25 + dense) consistently outperforms single-method retrieval.
From [Bruch et al. (2022)](https://arxiv.org/abs/2210.11934):

> "We study hybrid search in text retrieval where lexical and semantic search are
> fused together with the intuition that the two are complementary."

A dedicated fusion library lets you:
- **Zero dependencies**: Pure Rust, no runtime overhead
- **Algorithm flexibility**: RRF, CombMNZ, Borda, Weighted — swap easily
- **Database agnostic**: Works with any retriever's output

## When to Use Rank Fusion

Hybrid search combines multiple retrieval strategies (e.g., BM25 + dense vectors).
The challenge: different retrievers produce incompatible score scales.

| Retriever | Typical Score Range |
|-----------|---------------------|
| BM25/TF-IDF | 0 to 20+ (unbounded) |
| Cosine similarity | -1 to 1 |
| Dot product | varies wildly |

Rank fusion solves this by combining **ranks** or **normalized scores**.

## Quick Start

```rust
use rank_fusion::prelude::*;

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2), ("doc3", 8.0)];
let dense = vec![("doc2", 0.92), ("doc4", 0.85), ("doc1", 0.80)];

// Fuse with RRF (ignores scores, uses ranks only)
let fused = rrf(bm25, dense, RrfConfig::default());
// doc2 appears in both → ranked highest
```

## Prelude

Import common types with one line:

```rust
use rank_fusion::prelude::*;
// Imports: rrf, combsum, combmnz, borda, weighted
// Plus: FusionConfig, RrfConfig, WeightedConfig, FusionError, Result
```

## FusionMethod (Unified API)

Use the `FusionMethod` enum for a unified dispatch interface:

```rust
use rank_fusion::FusionMethod;

let sparse = vec![("d1", 10.0), ("d2", 8.0)];
let dense = vec![("d2", 0.9), ("d3", 0.7)];

// RRF
let fused = FusionMethod::Rrf { k: 60 }.fuse(&sparse, &dense);

// Weighted
let fused = FusionMethod::Weighted { w1: 0.3, w2: 0.7 }.fuse(&sparse, &dense);

// CombMNZ
let fused = FusionMethod::CombMnz.fuse(&sparse, &dense);
```

## Algorithms

| Function | Method | Uses Scores | Best For |
|----------|--------|-------------|----------|
| `rrf` | Reciprocal Rank Fusion | No | Different score scales |
| `rrf_weighted` | RRF with per-list weights | No | Heterogeneous retrievers |
| `combsum` | Sum of normalized scores | Yes | Same scale, trust scores |
| `combmnz` | Sum × overlap count | Yes | Reward multi-retriever agreement |
| `borda` | Borda count | No | Simple voting |
| `weighted` | Weighted combination | Yes | Trust one retriever more |

All have `*_multi` variants for 3+ lists.

### RRF k Parameter Tuning

The `k` parameter controls score distribution (default: 60, from Cormack et al., 2009):

| k Value | Effect | Use When |
|---------|--------|----------|
| 10-30 | Aggressive — top picks dominate | High confidence in retrievers |
| 50-100 | Balanced (default) | General hybrid search |
| 200+ | Conservative — rewards consensus | Diverse/experimental systems |

```rust
// Aggressive (trust top results)
let config = RrfConfig::new(20);

// Conservative (reward agreement)
let config = RrfConfig::new(100);

// Weighted RRF: trust dense 2x more than BM25
let weights = [0.33, 0.67];
let fused = rrf_weighted(&[&bm25, &dense], &weights, RrfConfig::default())?;
```

## Complete E2E Example

```rust
use rank_fusion::{rrf, combmnz, weighted, RrfConfig, WeightedConfig};

fn main() {
    // Simulated retrieval results from different sources
    let bm25_results = vec![
        ("doc_rust", 15.2),
        ("doc_python", 12.1),
        ("doc_go", 8.5),
    ];
    
    let dense_results = vec![
        ("doc_rust", 0.95),
        ("doc_cpp", 0.88),
        ("doc_python", 0.82),
    ];
    
    let keyword_results = vec![
        ("doc_rust", 3.0),
        ("doc_java", 2.5),
        ("doc_go", 2.0),
    ];
    
    // Method 1: RRF (best for incompatible scales)
    let rrf_fused = rrf(
        bm25_results.clone(),
        dense_results.clone(),
        RrfConfig::default().with_top_k(5),
    );
    println!("RRF top result: {:?}", rrf_fused.first());
    
    // Method 2: CombMNZ (rewards overlap)
    let combmnz_fused = combmnz(&bm25_results, &dense_results);
    println!("CombMNZ top result: {:?}", combmnz_fused.first());
    
    // Method 3: Weighted (trust dense more)
    let weighted_fused = weighted(
        &bm25_results,
        &dense_results,
        WeightedConfig::new(0.3, 0.7), // 30% BM25, 70% dense
    );
    println!("Weighted top result: {:?}", weighted_fused.first());
    
    // Method 4: Multi-list fusion (3+ sources)
    use rank_fusion::rrf_multi;
    let all_lists = vec![bm25_results, dense_results, keyword_results];
    let multi_fused = rrf_multi(&all_lists, RrfConfig::default());
    println!("Multi RRF top result: {:?}", multi_fused.first());
}
```

## Choosing an Algorithm

| Scenario | Recommendation |
|----------|----------------|
| BM25 + dense vectors | `rrf` — ignores incompatible scales |
| Same-scale scores, want overlap bonus | `combmnz` |
| You know one retriever is better | `weighted` with tuned weights |
| Simple, fast baseline | `borda` |
| 3+ retrievers | `*_multi` variants |

## Related

- [`rank-refine`](https://crates.io/crates/rank-refine) — Reranking algorithms (Matryoshka, ColBERT, cross-encoder)

## License

MIT OR Apache-2.0
