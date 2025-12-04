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

### Historical Context and Evolution

The development of rank fusion algorithms represents a gradual refinement of ideas spanning over two centuries, from voting theory to modern information retrieval. Understanding this evolution illuminates why certain methods exist and when to use them.

#### The Foundation: Voting Theory (1770-1785)

The problem of combining multiple rankings first emerged in political science, not computer science. When multiple voters rank candidates, how should we determine the collective preference?

**Borda Count (1770)**: Jean-Charles de Borda, a French mathematician and naval officer, proposed assigning points based on position: the top-ranked candidate gets N points, second gets N-1, and so on. This simple linear weighting scheme has a critical flaw: it's vulnerable to strategic voting. If voters know the system, they can manipulate outcomes by ranking strong opponents last. Despite this weakness, Borda's method established the fundamental principle that **position matters more than absolute scores**.

**Condorcet Method (1785)**: The Marquis de Condorcet, another French mathematician, recognized Borda's vulnerability and proposed pairwise comparisons instead. The Condorcet winner is the candidate who beats all others in head-to-head matchups. This is theoretically superior but has a practical problem: **Condorcet cycles** can occur (A beats B, B beats C, but C beats A), making a clear winner impossible. The modern **Kemeny optimal** ranking (1959) resolves this by finding the ranking that minimizes total disagreement, but computing it is NP-hard.

**The Lesson**: Early voting theory revealed a fundamental tension: simple methods are manipulable, optimal methods are intractable. Practical systems need approximations that balance robustness with computational feasibility.

#### The Information Retrieval Era: Score-Based Fusion (1994)

By the 1990s, information retrieval systems were producing ranked lists with relevance scores. Researchers at TREC (Text REtrieval Conference) faced a new problem: **combining results from multiple retrieval systems** that used different scoring functions.

**CombSUM (1994)**: Fox and Shaw's first approach was straightforward: sum the normalized scores from all systems. This worked when score scales were compatible, but normalization proved fragile. Different retrieval systems (BM25, TF-IDF, language models) produce scores with different distributions. Min-max normalization assumes uniform distributions, which rarely holds in practice. A document scoring 0.8 in one system might be exceptional, while 0.8 in another might be mediocre.

**CombMNZ (1994)**: Recognizing that documents appearing in multiple lists should be rewarded, Fox and Shaw multiplied CombSUM by the number of systems returning each document. This **multiplicity weighting** helps when systems agree, but amplifies CombSUM's normalization problems. If normalization is wrong, CombMNZ makes it worse.

**The Problem They Revealed**: Score-based fusion requires understanding score distributions. When distributions are unknown or vary by query, normalization becomes guesswork. TREC experiments (2001-2005) showed CombSUM and CombMNZ performed inconsistently: sometimes beating individual systems, sometimes underperforming. The variability came from normalization failures.

**Why They Persisted**: Despite their limitations, CombSUM and CombMNZ became baselines because they're simple and work when score scales are known. They represent the **optimistic approach**: assume scores are comparable, normalize carefully, and hope for the best.

#### The Breakthrough: Rank-Based Fusion (2009)

By 2009, the limitations of score-based fusion were well-documented. Cormack, Clarke, and Buettcher at the University of Waterloo proposed a radical simplification: **ignore scores entirely**.

**Reciprocal Rank Fusion (2009)**: RRF uses only rank positions, not scores. The formula `1/(k + rank)` has elegant properties:
- **No normalization needed**: Works with any score distribution because it ignores scores
- **Consensus reward**: Documents appearing high in multiple lists naturally score higher
- **Outlier resistance**: A single extreme score can't dominate because only position matters
- **Robustness**: Works consistently across different retrieval systems and datasets

**Why k=60?** Cormack et al. tested k values from 1 to 100 on TREC datasets. Lower k (e.g., k=10) overweights top positions, making the method sensitive to single-list agreement. Higher k (e.g., k=100) flattens the curve, reducing the advantage of top positions. k=60 provides the optimal balance: enough emphasis on top positions to reward quality, but enough flatness to reward consensus across lists.

**The Insight**: RRF represents the **pessimistic approach**: assume scores are incomparable, ignore them, and rely on rank consistency. This pessimism makes RRF robust where CombSUM is fragile.

**Empirical Validation**: TREC experiments showed RRF consistently outperformed individual retrieval systems and often matched or exceeded CombSUM when score scales were compatible. The key advantage: RRF works reliably even when score scales are unknown, while CombSUM requires careful tuning.

#### The Modern Era: Specialized Methods (2010-2024)

As hybrid search became common, new fusion methods emerged to address specific scenarios:

**DBSF (Distribution-Based Score Fusion)**: When you want to use scores but distributions differ, z-score normalization with clipping provides a middle ground. It's more robust than min-max but still requires score distributions to be approximately normal.

**Weighted Fusion**: When domain knowledge indicates one retriever is more reliable, weighted variants allow explicit trust modeling. This is particularly useful in production systems where retriever performance varies by query type.

**Additive Multi-Task Fusion**: E-commerce and recommendation systems need to optimize multiple objectives (CTR, conversion, revenue). ResFlow-style additive fusion allows explicit multi-task optimization without complex learning-to-rank models.

**The Pattern**: Each new method addresses a limitation of previous ones:
- CombSUM: Simple but fragile normalization
- CombMNZ: Adds consensus but amplifies normalization problems
- RRF: Robust but ignores score information
- DBSF: Uses scores but requires distribution knowledge
- Weighted: Adds domain knowledge but requires tuning

**Connection**: Rank fusion in IR is essentially "voting" where each retriever is a voter and documents are candidates. RRF's reciprocal formula ensures that documents appearing high in multiple lists (consensus) are favored, similar to how voting systems aggregate preferences. The evolution from Borda to RRF represents a shift from simple linear weighting to robust consensus-seeking, mirroring the same tension between simplicity and optimality that voting theory grappled with centuries earlier.

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

**Historical Origin**: Jean-Charles de Borda (1733-1799), a French mathematician, naval officer, and politician, proposed this method in 1770 for electing members to the French Academy of Sciences. Borda was frustrated by the Academy's voting system, which he believed could be manipulated. His method assigns points linearly: first place gets N points, second gets N-1, and so on.

**The Original Problem**: Borda's motivation was political. He observed that strategic voting could manipulate simple majority systems. If voters knew the system, they could rank strong opponents last to reduce their points. Borda's method reduces this manipulation by making the point difference between adjacent positions constant, but it's still vulnerable to strategic voting (the "Borda paradox").

**Why It Works for IR**: In information retrieval, we don't have strategic voters trying to manipulate outcomes. Retrievers are independent systems producing honest rankings. Borda's linear weighting scheme works well when all lists have similar lengths and we want to reward consistent high positions.

**Limitations**: Borda assumes all lists have equal importance and similar lengths. When lists differ significantly in length, the point differences become unbalanced. A document ranked 10th in a 100-item list gets 90 points, while a document ranked 1st in a 10-item list gets only 9 points. This makes Borda sensitive to list length variations.

**Modern Usage**: Borda remains useful as a baseline and works well when list lengths are similar. It's simpler than RRF but less robust to length variations. Each position gets points equal to how many documents it beats, making it intuitive but less sophisticated than reciprocal weighting.

### DBSF (Distribution-Based Score Fusion)

$$\text{DBSF}(d) = |R_d| \cdot \sum_{r \in R_d} \text{clip}\left(\frac{s_r(d) - \mu_r}{\sigma_r}, -3, 3\right)$$

where:
- $R_d$ is the set of retrievers containing document $d$
- $s_r(d)$ is the score of document $d$ from retriever $r$
- $\mu_r$ and $\sigma_r$ are the mean and standard deviation of scores from retriever $r$
- $\text{clip}(x, -3, 3)$ bounds $x$ to the range $[-3, 3]$

**Motivation**: DBSF addresses a limitation of CombSUM: min-max normalization assumes uniform distributions, which rarely holds. Z-score normalization (standardization) uses statistical properties (mean and standard deviation) to normalize scores, making it more robust to different distributions.

**Why Z-Score**: The z-score transformation `(x - μ) / σ` converts any distribution to a standard normal distribution (mean 0, standard deviation 1). This works when distributions are approximately normal, which is often true for relevance scores from well-tuned retrieval systems. Unlike min-max normalization, z-score preserves relative relationships within each distribution.

**Why Clipping**: In a normal distribution, 99.7% of values fall within ±3 standard deviations. Clipping to $[-3, 3]$ bounds outliers: a single extreme score can't dominate the fusion. Without clipping, a document with z-score 5.0 would have 5× the influence of a document with z-score 1.0, even though both are likely relevant. Clipping ensures no single score dominates while still using score information.

**The Trade-Off**: DBSF represents a middle ground between CombSUM (fragile normalization) and RRF (ignores scores). It uses scores (preserving information) but normalizes them robustly (addressing CombSUM's weakness). However, it requires score distributions to be approximately normal and requires computing statistics for each retriever, which adds complexity.

**When It Works**: DBSF works well when score distributions are known to be approximately normal and you want to use score information. It's more robust than CombSUM but less robust than RRF. The multiplicity weighting (like CombMNZ) rewards consensus across retrievers.

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
