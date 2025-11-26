# rank-fusion Design Document

## Context & Provenance

This crate emerged from research into the 2025 vector search landscape (HNSW, ScaNN, DiskANN, CAGRA, RaBitQ, ACORN, etc.) and the realization that the Rust ecosystem was missing clean, focused libraries for retrieval pipelines.

## The Two-Crate Strategy

The retrieval pipeline has distinct stages with different concerns:

```
Retrieve (BM25, Dense) → Fuse (rank-fusion) → Rerank (rank-refine) → Top-K
```

### rank-fusion (this crate)
- **Purpose:** Combine multiple result lists into one
- **Algorithms:** RRF, CombSUM, CombMNZ, Borda, Weighted
- **Philosophy:** Zero dependencies, pure algorithms, `no_std` compatible
- **Use case:** Hybrid search (BM25 + vectors), multi-index merging

### rank-refine (future crate)
- **Purpose:** Re-score a single list with expensive models
- **Planned features:**
  - Cross-encoder inference (via candle/ort)
  - Matryoshka tail refinement (SIMD-accelerated)
  - ColBERT/PLAID MaxSim operator
- **Philosophy:** Bring bleeding-edge reranking to Rust without Python

## Why These Names?

- "fusion" clearly describes combining lists (not model inference)
- "refine" suggests iterative improvement (the reranking step)
- Avoids the ambiguous "rerank" which means different things in IR

## Key Algorithms

### RRF (Reciprocal Rank Fusion)
- Formula: `score(d) = Σ 1/(k + rank)`
- Parameter-free (k=60 works universally)
- Ignores original scores, only uses rank position
- Robust to score distribution differences

### CombMNZ
- Sum of normalized scores × overlap count
- Rewards documents appearing in multiple lists
- Better than CombSUM when overlap is meaningful

### Borda Count
- Each position contributes `N - rank` points
- Classic voting theory adapted for IR
- Good for rank-only data (no scores)

## Performance Notes

- ~10M elements/sec on typical workloads (100-1000 items)
- `rrf_into` provides zero-allocation hot path
- HashMap-based accumulation, then sort
- Could be faster with arena allocation (future work)

## Future Directions

1. **Streaming fusion:** Process results as they arrive
2. **Top-K early termination:** Stop when top-K is stable
3. **SIMD score accumulation:** For very large lists
4. **Weighted RRF:** Per-source weighting

## References

- [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) (Cormack et al., 2009)
- [OneSparse](https://www.microsoft.com/en-us/research/publication/onesparse/) (Microsoft, 2024)
- [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) (2022)

