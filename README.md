# rank-fusion

Combine results from multiple retrieval systems into a single ranked list.

[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

## When to Use This

You have multiple retrieval sources returning ranked results:
- **BM25/keyword search** (Elasticsearch, Meilisearch, Tantivy)
- **Dense vector search** (embeddings from `sentence-transformers`, OpenAI, Cohere)
- **Sparse vectors** (SPLADE, BM25 as vectors)

You need to combine them into one ranked list. This crate does that.

## Quick Start

```toml
[dependencies]
rank-fusion = "0.1"
```

```rust
use rank_fusion::{rrf, RrfConfig};

// Results from BM25 search (doc_id, score)
let bm25 = vec![
    ("doc_42", 12.5),
    ("doc_17", 11.2),
    ("doc_89", 9.8),
];

// Results from vector search (doc_id, cosine_similarity)
let vectors = vec![
    ("doc_17", 0.92),
    ("doc_42", 0.87),
    ("doc_55", 0.85),
];

// Fuse with RRF (ignores scores, uses rank position only)
let fused = rrf(bm25, vectors, RrfConfig::default());
// doc_17 and doc_42 rank highest (appear in both lists)
```

## Algorithms

| Function | Best For | Uses Scores? |
|----------|----------|--------------|
| `rrf` | Default choice, different score scales | No |
| `combsum` | Similar score distributions | Yes |
| `combmnz` | Rewarding overlap | Yes |
| `borda` | Rank-only data | No |
| `weighted` | Trusting one source more | Yes |

### RRF (Reciprocal Rank Fusion)

**Use when:** Score distributions differ wildly (BM25 scores vs cosine similarity).

RRF ignores scores entirely — only rank position matters. This makes it robust when combining systems with incompatible scoring.

```rust
use rank_fusion::{rrf, RrfConfig};

let fused = rrf(bm25_results, vector_results, RrfConfig::default());
```

The `k` parameter (default 60) controls how much top ranks dominate:
- `k=1`: Top positions matter a lot
- `k=60`: Standard, balanced
- `k=100+`: More uniform contribution

### CombMNZ

**Use when:** Documents in multiple lists are more relevant.

```rust
use rank_fusion::combmnz;

let fused = combmnz(&bm25_results, &vector_results);
// Documents in both lists get 2× multiplier
```

### Weighted Fusion

**Use when:** You trust one retriever more than another.

```rust
use rank_fusion::{weighted, WeightedConfig};

// Trust vectors 70%, BM25 30%
let config = WeightedConfig::new(0.3, 0.7);
let fused = weighted(&bm25_results, &vector_results, config);
```

## Real-World Example: Hybrid Search with Qdrant

```rust
use rank_fusion::{rrf, RrfConfig};
use qdrant_client::prelude::*;

async fn hybrid_search(
    client: &QdrantClient,
    query_text: &str,
    query_vector: Vec<f32>,
) -> Vec<(String, f32)> {
    // 1. Dense vector search
    let dense_results = client
        .search_points(&SearchPoints {
            collection_name: "documents".into(),
            vector: query_vector,
            limit: 20,
            ..Default::default()
        })
        .await?;

    // 2. Sparse/keyword search (using Qdrant's sparse vectors or external)
    let sparse_results = keyword_search(query_text, 20).await?;

    // 3. Convert to (id, score) format
    let dense: Vec<_> = dense_results
        .result
        .iter()
        .map(|p| (p.id.to_string(), p.score))
        .collect();

    let sparse: Vec<_> = sparse_results
        .iter()
        .map(|(id, score)| (id.clone(), *score))
        .collect();

    // 4. Fuse with RRF
    rrf(dense, sparse, RrfConfig::default())
}
```

## Real-World Example: RAG Pipeline

```rust
use rank_fusion::{rrf_multi, RrfConfig};

struct RagPipeline {
    // Multiple retrievers for the same corpus
    dense_index: DenseIndex,      // e.g., nomic-embed-text-v1.5
    sparse_index: SparseIndex,    // e.g., SPLADE
    keyword_index: KeywordIndex,  // e.g., BM25
}

impl RagPipeline {
    async fn retrieve(&self, query: &str, top_k: usize) -> Vec<Document> {
        // Run all retrievers in parallel
        let (dense, sparse, keyword) = tokio::join!(
            self.dense_index.search(query, top_k * 2),
            self.sparse_index.search(query, top_k * 2),
            self.keyword_index.search(query, top_k * 2),
        );

        // Fuse all three
        let fused = rrf_multi(
            &[dense, sparse, keyword],
            RrfConfig::default(),
        );

        // Return top_k
        fused.into_iter()
            .take(top_k)
            .map(|(id, _)| self.get_document(&id))
            .collect()
    }
}
```

## Choosing an Algorithm

```
Do your retrievers have comparable score distributions?
├─ No  → Use RRF (scores ignored, robust)
└─ Yes → Do you want to reward documents in multiple lists?
         ├─ Yes → Use CombMNZ
         └─ No  → Do you trust one retriever more?
                  ├─ Yes → Use Weighted
                  └─ No  → Use CombSUM or RRF
```

## Performance

This crate has zero dependencies and runs in microseconds for typical workloads (100-1000 results per list). It's not the bottleneck in your retrieval pipeline.

## Next Steps: Reranking

After fusion, consider reranking the top results with a cross-encoder or other expensive model. See the companion crate [`rank-refine`](https://crates.io/crates/rank-refine).

```
Retrieve (BM25, vectors) → Fuse (this crate) → Rerank (rank-refine) → Top-K
```

## License

MIT OR Apache-2.0
