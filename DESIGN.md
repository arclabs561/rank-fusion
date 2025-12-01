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

### Historical Context

Rank fusion has roots in **social choice theory** (voting systems):

- **Borda Count (1770)**: Jean-Charles de Borda proposed N - rank points for election systems. Our `borda` function uses the same principle.

- **Condorcet Method (1785)**: Finds the ranking that beats all others in pairwise comparisons. The **Kemeny optimal** (NP-hard) is the modern formulation. RRF is a practical approximation.

- **Reciprocal Rank Fusion (2009)**: Cormack et al. introduced RRF for information retrieval, showing it outperforms individual rank learning methods. The k=60 default comes from their empirical studies on TREC datasets.

**Connection**: Rank fusion in IR is essentially "voting" where each retriever is a voter and documents are candidates. RRF's reciprocal formula ensures that documents appearing high in multiple lists (consensus) are favored, similar to how voting systems aggregate preferences.

## Algorithms

### Reciprocal Rank Fusion (RRF)

$$\text{RRF}(d) = \sum_{r \in R} \frac{1}{k + \text{rank}_r(d)}$$

where:
- $R$ is the set of all retrievers
- $k$ is a smoothing constant (default 60)
- $\text{rank}_r(d)$ is the 0-indexed rank of document $d$ in retriever $r$ (top result = 0)

**Properties:**
- Ignores score magnitudes — only position matters
- Outlier-resistant — a single high score doesn't dominate
- No normalization needed

**Why k=60?** Empirically chosen by Cormack et al. (2009) based on TREC evaluation. They tested k values from 1 to 100 and found k=60 provides the best balance between:
- Top position emphasis (rank 0 vs rank 5: 1.1x ratio)
- Consensus across lists (lower k overweights single-list agreement)
- Robustness across different datasets and retrieval systems

The effect of k on position weighting:
```
k=10:  rank 0 → 1/10 = 0.100    rank 5 → 1/15 = 0.067   (1.5x ratio)
k=60:  rank 0 → 1/60 = 0.017    rank 5 → 1/65 = 0.015   (1.1x ratio)
k=100: rank 0 → 1/100= 0.010    rank 5 → 1/105= 0.0095  (1.05x ratio)
```

Lower k → top ranks dominate. Higher k → flatter, rewards consensus across lists.

**When to tune k:**
- k=20-40: When top retrievers are highly reliable, want strong consensus
- k=60: Default for most hybrid search scenarios (works well across domains)
- k=100+: When lower-ranked items are still valuable, want broad agreement

### CombSUM / CombMNZ

$$\text{CombSUM}(d) = \sum_{r \in R} \text{norm}(s_r(d))$$

$$\text{CombMNZ}(d) = |R_d| \cdot \sum_{r \in R_d} \text{norm}(s_r(d))$$

where:
- $R$ is the set of all retrievers
- $R_d$ is the set of retrievers containing document $d$
- $s_r(d)$ is the score of document $d$ from retriever $r$
- $\text{norm}$ is min-max normalization (scales scores to [0, 1])

**When to use:** Scores are comparable (same scale, same meaning).

### Borda Count

$$\text{Borda}(d) = \sum_{r \in R} (N - \text{rank}_r(d))$$

where:
- $N$ is the total number of documents in list $r$
- $\text{rank}_r(d)$ is the 0-indexed rank of document $d$ in retriever $r$ (top result = 0)

**Origin:** Jean-Charles de Borda, 1770. Used in elections. Each position gets points equal to how many documents it beats.

### DBSF (Distribution-Based Score Fusion)

$$\text{DBSF}(d) = |R_d| \cdot \sum_{r \in R_d} \text{clip}\left(\frac{s_r(d) - \mu_r}{\sigma_r}, -3, 3\right)$$

where:
- $R_d$ is the set of retrievers containing document $d$
- $s_r(d)$ is the score of document $d$ from retriever $r$
- $\mu_r$ and $\sigma_r$ are the mean and standard deviation of scores from retriever $r$
- $\text{clip}(x, -3, 3)$ bounds $x$ to the range $[-3, 3]$

Z-score normalization puts different score distributions on comparable scales. Clipping to $[-3, 3]$ bounds outliers: in a normal distribution, 99.7% of values fall within ±3σ. This prevents a single extreme score from dominating the fusion.

## Optimal Solution

The theoretically optimal ranking is the **Kemeny optimal** — the ranking that minimizes total Kendall tau distance to all input rankings:

$$K^* = \arg\min_{K} \sum_{r \in R} \tau(K, r)$$

where $\tau(K, r)$ is the Kendall tau distance between ranking $K$ and input ranking $r$.

This is NP-hard to compute (shown to be NP-complete by Dwork et al., 2001). RRF is a practical approximation that achieves good results in O(n log n) time.

**Why RRF works well**: The reciprocal formula naturally rewards consensus (documents appearing in multiple lists) while being robust to score scale differences. Empirical studies show RRF performs within 2-5% of optimal methods on standard IR benchmarks while being orders of magnitude faster.

**When RRF underperforms**: OpenSearch benchmarks (BEIR dataset) show RRF is ~3-4% lower NDCG than CombSUM when score scales are compatible. This is the robustness vs. quality trade-off:
- RRF: Robust to scale mismatches, no tuning needed
- CombSUM: Better quality when scales match, requires normalization

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

## Failure Modes and Limitations

**When RRF underperforms:**
1. **Compatible score scales**: If all retrievers use the same scale (e.g., both cosine similarity 0-1), CombSUM with proper normalization achieves ~3-4% better NDCG (OpenSearch BEIR benchmarks).
2. **Very short lists**: When lists have <10 items, rank differences are less meaningful. Consider score-based methods.
3. **Highly correlated retrievers**: If retrievers are nearly identical, fusion adds little value. RRF still works but provides minimal benefit.

**When CombSUM/CombMNZ underperform:**
1. **Incompatible scales**: BM25 (0-100) vs dense (0-1) requires careful normalization. RRF avoids this entirely.
2. **Unknown distributions**: When score distributions are unknown or vary by query, normalization is fragile. RRF is more robust.
3. **Outlier scores**: A single extreme score can dominate CombSUM. RRF's rank-based approach is outlier-resistant.

**When to use what:**
- **RRF**: Default choice for hybrid search, unknown score scales, zero-configuration needs
- **CombSUM**: When scales are compatible and you can normalize reliably
- **CombMNZ**: When you want to reward overlap between lists
- **DBSF**: When score distributions differ but you want to use scores
- **Weighted**: When you have domain knowledge about retriever reliability

## Alternatives Considered

**Why not use a voting theory crate?**
- Overkill for rank fusion (we don't need full Condorcet, Schulze, etc.)
- Would add dependencies for little benefit
- Most voting theory crates focus on elections, not IR use cases

**Why not use HashMap from hashbrown?**
- std HashMap is fast enough for typical list sizes (<10K items)
- Avoiding deps keeps compile times minimal
- hashbrown's performance gains are marginal for our use case

## References

- Cormack, Clarke, Buettcher (2009). [Reciprocal Rank Fusion](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf)
- Fox & Shaw (1994). Combination of Multiple Searches
- Dwork et al. (2001). [Rank Aggregation Methods for the Web](https://dl.acm.org/doi/10.1145/371920.372165)
