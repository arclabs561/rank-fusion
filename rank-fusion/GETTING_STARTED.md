# Getting Started with rank-fusion

This guide walks you through using `rank-fusion` in real-world scenarios, from basic usage to complete RAG pipelines.

## Table of Contents

1. [Quick Start (5 minutes)](#quick-start-5-minutes)
2. [RAG Pipeline Example](#rag-pipeline-example)
3. [Hybrid Search Integration](#hybrid-search-integration)
4. [E-commerce Multi-Task Ranking](#e-commerce-multi-task-ranking)
5. [Python Integration](#python-integration)
6. [Debugging with Explainability](#debugging-with-explainability)

---

## Quick Start (5 minutes)

### Installation

```bash
cargo add rank-fusion
```

### Basic Example: Combining Two Retrievers

```rust
use rank_fusion::rrf;

// BM25 results (keyword search)
let bm25 = vec![
    ("doc_1", 87.5),
    ("doc_2", 82.3),
    ("doc_3", 78.1),
];

// Dense embedding results (semantic search)
let dense = vec![
    ("doc_2", 0.92),
    ("doc_1", 0.88),
    ("doc_4", 0.85),
];

// RRF finds consensus: doc_2 appears high in both lists
let fused = rrf(&bm25, &dense);
// Result: [("doc_2", 0.033), ("doc_1", 0.032), ("doc_3", 0.016), ("doc_4", 0.016)]
```

**Why RRF?** BM25 scores are 0-100, dense scores are 0-1. RRF ignores scores and uses only rank positions, so no normalization needed.

### Multiple Retrievers

```rust
use rank_fusion::rrf_multi;

let bm25 = vec![("doc_1", 87.5), ("doc_2", 82.3)];
let dense = vec![("doc_2", 0.92), ("doc_3", 0.88)];
let sparse = vec![("doc_1", 0.95), ("doc_4", 0.90)];

let fused = rrf_multi(&[&bm25[..], &dense[..], &sparse[..]], Default::default());
```

### Limiting Results

```rust
use rank_fusion::{rrf_with_config, RrfConfig};

let config = RrfConfig::default().with_top_k(10);
let fused = rrf_with_config(&bm25, &dense, config);
// Returns only top 10 results
```

---

## RAG Pipeline Example

Complete example: Elasticsearch → Fusion → Reranking → LLM

```rust
use rank_fusion::{rrf_multi, RrfConfig};

// In a real implementation, you would integrate with your search infrastructure:
// - Elasticsearch/OpenSearch for BM25 search
// - Vector database (Qdrant, Pinecone, etc.) for dense search
// - Cross-encoder model for reranking
//
// For integration examples, see INTEGRATION.md

fn rag_pipeline(query: &str) -> Vec<String> {
    // Step 1: Retrieve from multiple sources
    // Example: Replace with actual search calls
    let bm25_results: Vec<(String, f32)> = vec![
        ("doc_123".to_string(), 87.5),
        ("doc_456".to_string(), 82.3),
        ("doc_789".to_string(), 78.1),
    ];
    
    let dense_results: Vec<(String, f32)> = vec![
        ("doc_456".to_string(), 0.92),
        ("doc_123".to_string(), 0.88),
        ("doc_999".to_string(), 0.85),
    ];
    
    // Step 2: Fuse results (RRF finds consensus)
    let config = RrfConfig::default().with_top_k(100); // Top 100 for reranking
    let fused = rrf_multi(
        &[&bm25_results[..], &dense_results[..]],
        config,
    );
    
    // Step 3: Rerank with cross-encoder (expensive, so only top 100)
    // Example: Replace with actual reranking call
    let reranked: Vec<(String, f32)> = fused; // In real code: cross_encoder_rerank(&ids, query)
    
    // Step 4: Return top 10 for LLM context
    reranked.into_iter()
        .take(10)
        .map(|(id, _)| id)
        .collect()
}
```

### With Error Handling

```rust
use rank_fusion::{rrf_multi, RrfConfig};

fn rag_pipeline_safe(query: &str) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    // Step 1: Retrieve from multiple sources
    // In real code, these would return Result types from your search infrastructure
    let bm25_results: Vec<(String, f32)> = vec![
        ("doc_123".to_string(), 87.5),
        ("doc_456".to_string(), 82.3),
    ];
    
    let dense_results: Vec<(String, f32)> = vec![
        ("doc_456".to_string(), 0.92),
        ("doc_123".to_string(), 0.88),
    ];
    
    // Validate inputs
    if bm25_results.is_empty() && dense_results.is_empty() {
        return Ok(Vec::new());
    }
    
    // Step 2: Fuse results
    // Note: rrf_multi returns Vec, not Result - it handles empty lists gracefully
    let config = RrfConfig::default().with_top_k(100);
    let fused = rrf_multi(
        &[&bm25_results[..], &dense_results[..]],
        config,
    );
    
    if fused.is_empty() {
        return Ok(Vec::new());
    }
    
    // Step 3: Rerank (in real code, this would return Result)
    let reranked: Vec<(String, f32)> = fused; // Replace with actual reranking call
    
    // Step 4: Return top 10
    Ok(reranked.into_iter()
        .take(10)
        .map(|(id, _)| id)
        .collect())
}
```

---

## Hybrid Search Integration

### With Elasticsearch/OpenSearch

```rust
use rank_fusion::rrf;

// Elasticsearch BM25 query
let bm25_results: Vec<(String, f32)> = es_client
    .search(Query::new("match", query))
    .execute()
    .await?
    .hits
    .into_iter()
    .map(|hit| (hit.id, hit.score))
    .collect();

// Vector search (e.g., using OpenSearch k-NN)
let dense_results: Vec<(String, f32)> = es_client
    .search(Query::new("knn", query_embedding))
    .execute()
    .await?
    .hits
    .into_iter()
    .map(|hit| (hit.id, hit.score))
    .collect();

// Fuse results
let fused = rrf(&bm25_results, &dense_results);

// Return to user
fused.into_iter()
    .take(20)
    .map(|(id, _)| id)
    .collect()
```

### With Qdrant/Pinecone

```rust
use rank_fusion::rrf;

// BM25 from Elasticsearch
let bm25 = elasticsearch_search(query)?;

// Dense from Qdrant
let dense: Vec<(String, f32)> = qdrant_client
    .search(query_embedding)
    .await?
    .into_iter()
    .map(|point| (point.id, point.score))
    .collect();

// Fuse
let fused = rrf(&bm25, &dense);
```

---

## E-commerce Multi-Task Ranking

Combine CTR (click-through rate) and CTCVR (click-to-conversion rate) predictions:

```rust
use rank_fusion::{additive_multi_task_with_config, AdditiveMultiTaskConfig, Normalization};

// CTR predictions (0.0 to 1.0)
let ctr_scores = vec![
    ("product_123", 0.15),
    ("product_456", 0.12),
    ("product_789", 0.10),
];

// CTCVR predictions (0.0 to 1.0, typically lower than CTR)
let ctcvr_scores = vec![
    ("product_123", 0.08),
    ("product_456", 0.06),
    ("product_789", 0.05),
];

// ResFlow-style: CTR + CTCVR × 20
// This weights conversion 20x more than clicks
let config = AdditiveMultiTaskConfig::new((1.0, 20.0))
    .with_normalization(Normalization::MinMax);

let ranked = additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config);
// product_123 wins: 0.15 + 0.08×20 = 1.75
// product_456 second: 0.12 + 0.06×20 = 1.32
```

### A/B Testing Setup

```rust
use rank_fusion::{rrf_explain, RetrieverId, RrfConfig};
use rank_fusion::explain::analyze_consensus;

// Control: RRF with k=60
let control_config = RrfConfig::new(60);
let control_results = rrf_explain(
    &[&bm25[..], &dense[..]],
    &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
    control_config,
);

// Variant: RRF with k=40 (stronger consensus required)
let variant_config = RrfConfig::new(40);
let variant_results = rrf_explain(
    &[&bm25[..], &dense[..]],
    &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
    variant_config,
);

// Analyze consensus patterns
let control_consensus = analyze_consensus(&control_results);
let variant_consensus = analyze_consensus(&variant_results);

// Compare: variant should have more high_consensus items
println!("Control high consensus: {}", control_consensus.high_consensus.len());
println!("Variant high consensus: {}", variant_consensus.high_consensus.len());
```

---

## Python Integration

### Installation

```bash
pip install rank-fusion
```

### Basic Usage

```python
import rank_fusion

# BM25 results
bm25 = [("doc_1", 87.5), ("doc_2", 82.3), ("doc_3", 78.1)]

# Dense results
dense = [("doc_2", 0.92), ("doc_1", 0.88), ("doc_4", 0.85)]

# RRF fusion
fused = rank_fusion.rrf(bm25, dense, k=60)
# [("doc_2", 0.033), ("doc_1", 0.032), ...]

# With top_k
fused_top10 = rank_fusion.rrf(bm25, dense, k=60, top_k=10)
```

### All Algorithms Available

```python
import rank_fusion

# Rank-based (ignores scores)
fused = rank_fusion.rrf(bm25, dense, k=60)
fused = rank_fusion.isr(bm25, dense, k=1)
fused = rank_fusion.borda(bm25, dense)

# Score-based (uses scores)
fused = rank_fusion.combsum(bm25, dense)
fused = rank_fusion.combmnz(bm25, dense)
fused = rank_fusion.dbsf(bm25, dense)
fused = rank_fusion.weighted(bm25, dense, weight_a=0.7, weight_b=0.3)

# Multiple lists
fused = rank_fusion.rrf_multi([bm25, dense, sparse], k=60)
fused = rank_fusion.combsum_multi([bm25, dense, sparse])

# E-commerce
fused = rank_fusion.additive_multi_task(
    ctr_scores, ctcvr_scores,
    weights=(1.0, 20.0),
    normalization="minmax"
)
```

### Explainability in Python

```python
import rank_fusion

# Get explanations
explained = rank_fusion.rrf_explain(
    [bm25, dense],
    ["bm25", "dense"],
    k=60
)

# Inspect first result
result = explained[0]
print(f"Document: {result.id}")
print(f"Score: {result.score}")
print(f"Consensus: {result.explanation.consensus_score}")
print(f"Sources: {len(result.explanation.sources)}")

# See which retrievers contributed
for source in result.explanation.sources:
    print(f"  {source.retriever_id}: rank={source.original_rank}, contribution={source.contribution}")
```

---

## Debugging with Explainability

### Problem: Why did this document rank so low?

```rust
use rank_fusion::explain::{rrf_explain, RetrieverId};
use rank_fusion::RrfConfig;

let explained = rrf_explain(
    &[&bm25[..], &dense[..]],
    &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
    RrfConfig::default(),
);

// Find a specific document
if let Some(result) = explained.iter().find(|r| r.id == "doc_123") {
    println!("Document: {}", result.id);
    println!("Final rank: {}", result.rank);
    println!("Final score: {}", result.score);
    println!("Consensus: {}%", result.explanation.consensus_score * 100.0);
    
    for source in &result.explanation.sources {
        println!("  {}: rank={:?}, score={:?}, contribution={}",
            source.retriever_id,
            source.original_rank,
            source.original_score,
            source.contribution
        );
    }
}
```

### Problem: Are my retrievers agreeing?

```rust
use rank_fusion::explain::{rrf_explain, analyze_consensus, RetrieverId};
use rank_fusion::RrfConfig;

let explained = rrf_explain(
    &[&bm25[..], &dense[..]],
    &[RetrieverId::new("bm25"), RetrieverId::new("dense")],
    RrfConfig::default(),
);

let consensus = analyze_consensus(&explained);

// Documents in all retrievers (strong consensus)
println!("High consensus: {:?}", consensus.high_consensus);

// Documents only in one retriever (potential gaps)
println!("Single source: {:?}", consensus.single_source);

// Documents with large rank disagreements
for (doc_id, rank_info) in &consensus.rank_disagreement {
    println!("{}: {:?}", doc_id, rank_info);
    // Example: doc_123: [("bm25", 0), ("dense", 50)]
    // This document is rank 0 in BM25 but rank 50 in dense - large disagreement!
}
```

---

## Performance Considerations

### Latency

- **RRF for 100 items**: ~13μs (suitable for real-time)
- **RRF for 1000 items**: ~159μs
- **RRF for 5 lists × 100 items**: ~38μs

### Memory

- Pre-allocates hash maps with estimated capacity
- No allocations during fusion (except final Vec)
- Zero-copy when possible (uses slices)

### When to Use `top_k`

Always use `top_k` when you only need the top N results:

```rust
// Good: Only compute top 10
let config = RrfConfig::default().with_top_k(10);
let fused = rrf_with_config(&bm25, &dense, config);

// Less efficient: Compute all, then truncate
let fused = rrf(&bm25, &dense);
let top10 = fused.into_iter().take(10).collect::<Vec<_>>();
```

---

## Next Steps

- **See [API Documentation](https://docs.rs/rank-fusion)** for all functions
- **See [DESIGN.md](DESIGN.md)** for algorithm details
- **See [INTEGRATION.md](INTEGRATION.md)** for framework-specific examples
- **See [examples/](examples/)** for complete runnable examples

---

## Common Pitfalls

### 1. Using RRF when scores are compatible

**Problem**: Using RRF when both retrievers use the same scale (e.g., both cosine similarity 0-1).

**Solution**: Use `combsum` or `combmnz` for better quality:

```rust
// Both are cosine similarity (0-1 scale)
let dense1 = vec![("doc_1", 0.9), ("doc_2", 0.8)];
let dense2 = vec![("doc_2", 0.92), ("doc_3", 0.85)];

// Use CombSUM (better quality when scales match)
let fused = combsum(&dense1, &dense2);
```

### 2. Not validating k >= 1

**Problem**: Passing `k=0` causes division by zero (returns empty result).

**Solution**: Always use `k >= 1`:

```rust
// ❌ Bad
let config = RrfConfig::new(0); // Returns empty result

// ✅ Good
let config = RrfConfig::new(60); // Default
```

### 3. Forgetting to handle empty lists

**Problem**: Empty input lists return empty results, which might break downstream code.

**Solution**: Validate inputs:

```rust
if bm25_results.is_empty() && dense_results.is_empty() {
    return Ok(Vec::new());
}
```

---

## Getting Help

- **GitHub Issues**: https://github.com/arclabs561/rank-fusion/issues
- **Documentation**: https://docs.rs/rank-fusion
- **Examples**: See `examples/` directory

