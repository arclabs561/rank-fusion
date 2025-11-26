# Design

Two crates for retrieval pipelines:

```
Retrieve → Fuse (this crate) → Refine (rank-refine) → Top-K
```

**rank-fusion** — combine ranked lists, zero deps
**rank-refine** — expensive re-scoring (cross-encoder, Matryoshka, ColBERT)

## Algorithms

| Function | Method | Uses Scores |
|----------|--------|-------------|
| `rrf` | 1/(k+rank) | No |
| `combsum` | sum(norm(scores)) | Yes |
| `combmnz` | sum × count | Yes |
| `borda` | N-rank voting | No |
| `weighted` | w·norm(score) | Yes |

## References

- [RRF paper](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — Cormack et al. 2009
- Fox & Shaw 1994 — CombSUM, CombMNZ
