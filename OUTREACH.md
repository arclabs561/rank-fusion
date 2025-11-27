# Outreach: Integration Opportunities

## langchain-rust

**Repository**: https://github.com/Abraxas-365/langchain-rust

**Issue Title**: Feature: Hybrid search fusion (RRF, CombMNZ)

### Summary

Add rank fusion for combining results from multiple retrievers (e.g., BM25 + dense embeddings).

### Use Case

Hybrid search is a common pattern where you want to combine:
- Lexical search (BM25, keyword matching)
- Semantic search (dense embeddings)
- Sparse vectors (SPLADE)

Currently users have to implement fusion themselves. A built-in solution would make hybrid search much easier.

### Proposed Solution

Consider integrating [rank-fusion](https://crates.io/crates/rank-fusion) which provides:

| Algorithm | Use Case |
|-----------|----------|
| `rrf` | Different score scales (BM25 + cosine) |
| `combsum` | Same score scales |
| `combmnz` | Reward overlap between retrievers |
| `dbsf` | Different score distributions (z-score) |

Example API:

```rust
use langchain_rust::retriever::HybridRetriever;
use rank_fusion::rrf;

let bm25_results = bm25_retriever.search(&query).await?;
let dense_results = dense_retriever.search(&query).await?;

let fused = rrf(&bm25_results, &dense_results);
```

The crate is zero-dependency, MIT/Apache-2.0, and benchmarks at ~13μs for 100 items.

---

## swiftide

**Repository**: https://github.com/bosun-ai/swiftide

**Issue Title**: Feature: Diversity selection for query results (MMR, DPP)

### Summary

Add diversity selection algorithms (MMR, DPP) for query pipeline results to reduce redundancy.

### Problem

When retrieving documents for RAG, the top-k results often contain near-duplicates or highly similar content. This wastes context window and reduces answer quality.

### Proposed Solution

Add a `DiversityTransformer` or similar step for query pipelines that reranks results to balance relevance with diversity.

Consider integrating [rank-refine](https://crates.io/crates/rank-refine) which provides:
- `mmr_cosine(candidates, embeddings, config)` 
- `dpp(candidates, embeddings, config)`
- SIMD-accelerated similarity computation
- Zero mandatory dependencies

Example usage:

```rust
query::Pipeline::from_search(qdrant)
    .then(DiversitySelector::mmr(lambda: 0.7))
    .then(ContextBuilder::default())
    .then(Answer::simple())
```

---

## Status

- [ ] langchain-rust: Create issue manually
- [ ] swiftide: Create issue manually
- [ ] fastembed-rs: Consider PR for MaxSim scoring (they already have reranking via ONNX)

## Notes

- qdrant already has full ColBERT/MaxSim support (completed July 2024)
- fastembed-rs focuses on model inference, not scoring—different layer

