# Design

## Dependencies

**Zero.** Pure Rust standard library.

| Metric | Value |
|--------|-------|
| Compile time | ~0.1s |
| Binary size | ~10KB |
| Transitive deps | 0 |

Why: The algorithms are simple. External crates add overhead without benefit.

## Problem Space

Given $n$ ranked lists from different retrievers, produce one combined ranking.

This is the **rank aggregation** problem from social choice theory — the same mathematics behind voting systems (Condorcet, Borda) and sports rankings (Elo).

Key insight: **We don't look at document content.** Only scores and positions.

## Algorithms

### Reciprocal Rank Fusion (RRF)

```math
\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}
```

where $k$ is a smoothing constant (default 60).

**Properties:**
- Ignores score magnitudes — only position matters
- Outlier-resistant — a single high score doesn't dominate
- No normalization needed

**Why k=60?** Empirically chosen by Cormack et al. (2009) to balance:
- Low k → top positions dominate
- High k → positions matter less, rewards consensus

### CombSUM / CombMNZ

```math
\text{CombSUM}(d) = \sum_{r \in R} \text{norm}(s_r(d))
```

```math
\text{CombMNZ}(d) = |R_d| \cdot \sum_{r \in R_d} \text{norm}(s_r(d))
```

where $R_d$ is the set of lists containing $d$, and $\text{norm}$ is min-max normalization.

**When to use:** Scores are comparable (same scale, same meaning).

### Borda Count

```math
\text{Borda}(d) = \sum_{r \in R} (N - \text{rank}_r(d))
```

**Origin:** Jean-Charles de Borda, 1770. Used in elections.

### DBSF (Distribution-Based Score Fusion)

```math
\text{DBSF}(d) = |R_d| \cdot \sum_{r \in R_d} \text{clip}\left(\frac{s_r(d) - \mu_r}{\sigma_r}, -3, 3\right)
```

Z-score normalization with clipping. More robust than min-max when distributions differ.

## Optimal Solution

The theoretically optimal ranking is the **Kemeny optimal** — the ranking that minimizes total Kendall tau distance to all input rankings:

```math
K^* = \arg\min_{K} \sum_{r \in R} \tau(K, r)
```

This is NP-hard to compute. RRF is a practical approximation.

## Complexity

| Algorithm | Time | Space |
|-----------|------|-------|
| RRF | O(n log n) | O(n) |
| CombSUM | O(n log n) | O(n) |
| Borda | O(n log n) | O(n) |

All dominated by the final sort.

## API Design

### Generic IDs

Functions accept any `I: Clone + Eq + Hash`:

```rust
rrf::<&str>(&[("doc1", 0.9)], &[("doc2", 0.8)])
rrf::<u64>(&[(1, 0.9)], &[(2, 0.8)])
```

### Two-List vs Multi-List

```rust
rrf(&a, &b)           // optimized, no Vec allocation
rrf_multi(&[a, b, c]) // general case
```

## Alternatives Considered

**Why not use a voting theory crate?**
- Overkill for rank fusion (we don't need full Condorcet, Schulze, etc.)
- Would add dependencies for little benefit

**Why not use HashMap from hashbrown?**
- std HashMap is fast enough for typical list sizes (<10K items)
- Avoiding deps keeps compile times minimal

## References

- Cormack, Clarke, Buettcher (2009). [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Fox & Shaw (1994). Combination of Multiple Searches
- Dwork et al. (2001). [Rank Aggregation Methods for the Web](https://dl.acm.org/doi/10.1145/371920.372165)
