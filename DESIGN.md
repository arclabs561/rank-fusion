# rank-fusion

Rank fusion algorithms for combining retrieval results.

## Context

This crate is part of a two-crate approach to retrieval pipelines:

```
Retrieve → Fuse (this crate) → Refine (rank-refine) → Top-K
```

**rank-fusion** combines multiple result lists into one. Zero dependencies.

**rank-refine** (sibling crate) re-scores with expensive models (cross-encoder, Matryoshka, ColBERT). Has ML dependencies.

## Algorithms

| Function | Method | Uses Scores? |
|----------|--------|--------------|
| `rrf` | Reciprocal Rank Fusion | No (rank only) |
| `combsum` | Sum of normalized scores | Yes |
| `combmnz` | Sum × overlap count | Yes |
| `borda` | Borda count voting | No (rank only) |
| `weighted` | Weighted combination | Yes |

### When to use what

- **RRF**: Default choice. Robust when score distributions differ.
- **CombMNZ**: When overlap is meaningful (doc in both lists = more relevant).
- **Borda**: When you only have ranks, no scores.
- **Weighted**: When you trust one source more than another.

## Implementation Notes

- HashMap-based score accumulation, then sort
- `rrf_into` reuses output buffer for hot paths
- ~10M elements/sec on typical workloads (100-1000 items)

## References

- Cormack et al. (2009) — [Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Fox & Shaw (1994) — Combination of Multiple Searches (CombSUM, CombMNZ)
