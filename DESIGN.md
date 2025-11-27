# Design

## Mathematical Foundation

Three distinct problems arise in retrieval pipelines. Each belongs to a different mathematical field:

| Problem | Field | Input | Output | Crate |
|---------|-------|-------|--------|-------|
| **Aggregation** | Social Choice Theory | n rankings | 1 ranking | rank-fusion |
| **Scoring** | Learning to Rank | (query, doc) pairs | scores | rank-refine |
| **Selection** | Submodular Optimization | set + candidates | diverse subset | rank-refine |

### 1. Rank Aggregation (This Crate)

**Problem**: Given n ranked lists from different retrievers, produce one consensus ranking.

This is the [rank aggregation problem](https://en.wikipedia.org/wiki/Rank_aggregation) from social choice theory. The same mathematics underlies:
- Voting systems (Condorcet, Borda, 18th century)
- Sports rankings (Elo, Glicko)
- Metasearch engines (2000s)

**Key insight**: We don't look at document content. Only scores and ranks.

| Algorithm | Origin | Key Property |
|-----------|--------|--------------|
| Borda | Jean-Charles de Borda, 1770 | Position-based scoring |
| RRF | Cormack et al., 2009 | Outlier-resistant, scale-agnostic |
| CombSUM/MNZ | Fox & Shaw, 1994 | Score combination with overlap bonus |
| DBSF | Weaviate, 2023 | Z-score normalization |

**Optimal solution**: The [Kemeny optimal ranking](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method) minimizes Kendall tau distance to all input rankings. It's NP-hard to compute, so we use approximations like RRF.

### 2. Scoring / Reranking (rank-refine)

**Problem**: Given a query and candidate documents, compute relevance scores.

This is [learning to rank](https://en.wikipedia.org/wiki/Learning_to_rank) — a machine learning problem. Methods differ in what they model:

| Method | Input | Complexity | Quality |
|--------|-------|------------|---------|
| Dense (dot/cosine) | 1 vec × 1 vec | O(d) | Good |
| Late Interaction (MaxSim) | 1 vec × n tokens | O(q × d) | Better |
| Cross-encoder | text pair | O(n) inference | Best |

**Key insight**: We look at document content (embeddings). Scoring is a function f(query, doc) → ℝ.

### 3. Diversity Selection (rank-refine.diversity)

**Problem**: Given a scored candidate set, select a diverse subset that balances relevance and variety.

This is [submodular optimization](https://en.wikipedia.org/wiki/Submodular_set_function). The mathematical structure:

```
f(S ∪ {x}) - f(S) ≤ f(T ∪ {x}) - f(T)  for T ⊆ S
```

Selecting an item has **diminishing returns** as more items are already selected.

| Algorithm | Formulation | Properties |
|-----------|-------------|------------|
| MMR | argmax λ·rel(d) - (1-λ)·max_s sim(d,s) | Greedy, O(n×k) |
| DPP | P(S) ∝ det(L_S) | Probabilistic, expensive |
| Facility Location | max Σ_i max_{j∈S} sim(i,j) | Coverage-optimal |

**Key insight**: We look at inter-item similarity. Selection is a set function f(S) → ℝ, not pairwise.

## Why Two Crates, Not Three?

We considered a separate `rank-select` crate for diversity. Arguments against:

1. **Code size**: MMR is ~150 lines. Not enough to justify a crate.
2. **Shared infrastructure**: Diversity uses the same SIMD cosine from rank-refine.
3. **Usage pattern**: Users typically rerank then diversify in the same pipeline.

The mathematical distinction is clear, but the practical separation isn't justified.

## Pipeline Position

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           RETRIEVAL PIPELINE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐     ┌───────────┐ │
│  │  Retrieval  │     │   Fusion    │     │  Reranking  │     │ Selection │ │
│  │   (ANN)     │────▶│ rank-fusion │────▶│ rank-refine │────▶│   (MMR)   │ │
│  │  millions   │     │  combine    │     │  rescore    │     │ diversify │ │
│  └─────────────┘     └─────────────┘     └─────────────┘     └───────────┘ │
│        ↓                   ↓                   ↓                   ↓       │
│     1000s              100-500              50-100               10-20     │
│                                                                             │
│  Problem:           Aggregation          Scoring              Selection    │
│  Field:             Social Choice        Learning to Rank     Submodular   │
│  Content:           No                   Yes                  Yes (inter)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

## API Design Principles

### 1. Scores vs Ranks

Some algorithms use scores, others only ranks:

```rust
// Score-based: preserves magnitude information
combsum(&[(d1, 0.9), (d2, 0.1)], &[(d2, 0.8), (d3, 0.2)])

// Rank-based: only position matters
rrf(&[(d1, 0.9), (d2, 0.1)], &[(d2, 0.8), (d3, 0.2)])
// Score values ignored; d1 is rank 1, d2 is rank 2
```

Rank-based methods (RRF, Borda) are preferred when scores are incomparable.

### 2. Generic over ID Type

All functions are generic over `I: Clone + Eq + Hash`:

```rust
rrf::<&str>(&[("doc1", 0.9)], &[("doc2", 0.8)])  // String IDs
rrf::<u64>(&[(1, 0.9)], &[(2, 0.8)])              // Integer IDs
rrf::<Uuid>(&[(uuid1, 0.9)], &[(uuid2, 0.8)])    // Custom IDs
```

### 3. Two-List vs Multi-List

Most algorithms have both variants:

```rust
rrf(&a, &b)           // Two lists (common case)
rrf_multi(&[a, b, c]) // Three+ lists
```

Two-list is optimized separately (no Vec allocation for list references).

## Algorithms

| Function | Method | Uses Scores | Best For |
|----------|--------|-------------|----------|
| `rrf` | 1/(k+rank) | No | Incompatible score scales |
| `combsum` | sum(norm(scores)) | Yes | Similar scales, trust scores |
| `combmnz` | sum × count | Yes | Reward overlap between lists |
| `borda` | N-rank voting | No | Simple voting |
| `weighted` | w·norm(score) | Yes | Custom retriever weights |
| `dbsf` | sum(z-score) | Yes | Different score distributions |

## References

### Rank Aggregation (Social Choice)
- Condorcet, 1785 — Pairwise majority preferences
- Borda, 1770 — Positional scoring
- [Kemeny, 1959](https://en.wikipedia.org/wiki/Kemeny%E2%80%93Young_method) — Optimal aggregation (NP-hard)
- [Dwork et al., 2001](https://dl.acm.org/doi/10.1145/371920.372165) — Rank aggregation methods

### Fusion in IR
- [Fox & Shaw, 1994](https://dl.acm.org/doi/10.1145/188490.188561) — CombSUM, CombMNZ
- [Cormack et al., 2009](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF
- [Bruch et al., 2022](https://arxiv.org/abs/2210.11934) — An analysis of fusion functions

### Submodular Optimization
- [Nemhauser et al., 1978](https://link.springer.com/article/10.1007/BF01588971) — Greedy submodular maximization
- [Carbonell & Goldstein, 1998](https://dl.acm.org/doi/10.1145/290941.291025) — MMR
