# Decision Guides

Quick reference guides for choosing algorithms, parameters, and configurations.

## Which Fusion Algorithm?

```
┌─────────────────────────────────────────────────────────┐
│ Do you know the score scales are compatible?           │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ NO                           │ YES
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use RRF          │         │ Do you want to reward     │
│ (rank-based)     │         │ documents appearing in    │
│                  │         │ multiple lists?           │
│ • No tuning      │         └──────────────────────────┘
│ • Works always   │                    │
│ • k=60 default   │                    │
└──────────────────┘                    │
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    │ YES                                   │ NO
                    │                                       │
                    ▼                                       ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │ Use CombMNZ      │                  │ Use CombSUM      │
         │                  │                  │                  │
         │ • Multiplies by  │                  │ • Simple sum     │
         │   overlap count  │                  │ • Fast           │
         │ • Rewards        │                  │ • Default choice  │
         │   consensus     │                  │   for same scale │
         └──────────────────┘                  └──────────────────┘
```

### Detailed Comparison

| Algorithm | Score Scales | Tuning Required | Best For |
|-----------|--------------|-----------------|----------|
| **RRF** | Any (ignores scores) | None (k=60 works) | Hybrid search, unknown scales |
| **ISR** | Any (ignores scores) | None (k=1 works) | When lower ranks matter more |
| **CombSUM** | Compatible | Normalization | Same embedding model, trusted scores |
| **CombMNZ** | Compatible | Normalization | Want to reward overlap |
| **Borda** | Any (ignores scores) | None | Simple voting, similar list lengths |
| **DBSF** | Different distributions | None (auto z-score) | Known distributions, want scores |
| **Standardized** | Different distributions | Clip range | Robust to outliers (ERANK-style) |
| **Weighted** | Compatible | Weight selection | Domain knowledge about reliability |
| **Additive Multi-Task** | Compatible | Task weights | E-commerce (CTR + CTCVR) |

## RRF k Parameter Selection

```
┌─────────────────────────────────────────────────────────┐
│ How reliable are your top retrievers?                    │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ HIGHLY RELIABLE              │ UNKNOWN/MIXED
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ k = 20-40        │         │ k = 60 (default)         │
│                  │         │                           │
│ • Strong         │         │ • Balanced                │
│   consensus      │         │ • Works across domains    │
│   required       │         │ • Empirically validated   │
│ • Top positions  │         │   (Cormack et al., 2009) │
│   dominate       │         │                           │
└──────────────────┘         └──────────────────────────┘
                                        │
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    │ LOWER RANKS STILL VALUABLE            │
                    │                                       │
                    ▼                                       ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │ k = 100+         │                  │ k = 60           │
         │                  │                  │ (already set)   │
         │ • Flatter curve  │                  │                  │
         │ • Broad          │                  │                  │
         │   agreement      │                  │                  │
         │ • Less emphasis  │                  │                  │
         │   on top         │                  │                  │
         └──────────────────┘                  └──────────────────┘
```

**Effect of k on position weighting:**
- k=10: rank 0 → 0.100, rank 5 → 0.067 (1.5× ratio) - top dominates
- k=60: rank 0 → 0.017, rank 5 → 0.015 (1.1× ratio) - balanced
- k=100: rank 0 → 0.010, rank 5 → 0.0095 (1.05× ratio) - flat

## CombSUM vs CombMNZ

```
┌─────────────────────────────────────────────────────────┐
│ Do documents appearing in multiple lists indicate       │
│ higher relevance?                                       │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use CombMNZ     │         │ Use CombSUM              │
│                  │         │                          │
│ • Multiplies by │         │ • Simple sum             │
│   overlap count │         │ • Faster                 │
│ • Rewards       │         │ • Less sensitive to      │
│   consensus     │         │   normalization errors   │
│ • Can amplify   │         │                          │
│   normalization │         │                          │
│   errors        │         │                          │
└──────────────────┘         └──────────────────────────┘
```

**Historical Note**: TREC experiments (2001-2005) showed CombSUM consistently outperformed CombMNZ, but CombMNZ sometimes exceeded individual systems when normalization was correct. The choice depends on normalization reliability.

## Normalization Strategy

```
┌─────────────────────────────────────────────────────────┐
│ Do you know the score distributions?                    │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Are distributions│         │ Use RRF                  │
│ approximately    │         │ (no normalization)       │
│ normal?          │         │                          │
└──────────────────┘         └──────────────────────────┘
         │                              │
         │                              │
    ┌────┴────┐                         │
    │         │                         │
    │ YES     │ NO                      │
    │         │                         │
    ▼         ▼                         │
┌────────┐ ┌────────┐                   │
│ Z-score│ │ Min-max│                   │
│ (DBSF) │ │        │                   │
│        │ │        │                   │
│ • Auto │ │ • Fast │                   │
│   stats│ │ • Works│                   │
│ • Clip │ │   when│                   │
│   [-3,3]│ │   range│                  │
│        │ │   known│                   │
└────────┘ └────────┘                   │
                                        │
                                        │
                    ┌───────────────────┴───────────────────┐
                    │                                       │
                    │ OUTLIERS A CONCERN?                  │
                    │                                       │
                    ▼                                       ▼
         ┌──────────────────┐                  ┌──────────────────┐
         │ Use Standardized  │                  │ Use DBSF         │
         │ (ERANK-style)     │                  │                  │
         │                   │                  │ • Z-score        │
         │ • Z-score + clip  │                  │ • No clipping     │
         │ • Robust to       │                  │ • Faster          │
         │   outliers        │                  │                  │
         │ • Configurable    │                  │                  │
         │   clip range      │                  │                  │
         └──────────────────┘                  └──────────────────┘
```

## Weighted Fusion Weight Selection

```
┌─────────────────────────────────────────────────────────┐
│ Do you have domain knowledge about retriever reliability?│
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use Weighted     │         │ Use RRF or CombSUM        │
│                  │         │ (equal weights)          │
│ • Set weights    │         │                          │
│   based on:      │         │                          │
│   - Past         │         │                          │
│     performance  │         │                          │
│   - Query type   │         │                          │
│   - Domain       │         │                          │
│     expertise    │         │                          │
│                  │         │                          │
│ Example:         │         │                          │
│ BM25: 0.3        │         │                          │
│ Dense: 0.7       │         │                          │
│ (dense is more   │         │                          │
│  reliable)       │         │                          │
└──────────────────┘         └──────────────────────────┘
```

**Weight Selection Guidelines:**
- Weights should sum to 1.0 (will be normalized automatically)
- Higher weight = more trusted retriever
- Start with equal weights, tune based on evaluation
- Consider query-dependent weights (e.g., BM25 for keyword queries, dense for semantic)

## Additive Multi-Task Fusion

```
┌─────────────────────────────────────────────────────────┐
│ Are you optimizing multiple objectives?                 │
│ (e.g., CTR + CTCVR in e-commerce)                      │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use Additive     │         │ Use single-task fusion    │
│ Multi-Task       │         │ (RRF, CombSUM, etc.)      │
│                  │         │                          │
│ • ResFlow-style  │         │                          │
│ • Configurable   │         │                          │
│   task weights   │         │                          │
│ • Multiple       │         │                          │
│   normalizations │         │                          │
│                  │         │                          │
│ Example:         │         │                          │
│ CTR: 1.0         │         │                          │
│ CTCVR: 20.0      │         │                          │
│ (conversion more │         │                          │
│  important)      │         │                          │
└──────────────────┘         └──────────────────────────┘
```

**Task Weight Guidelines:**
- Weights reflect relative importance of tasks
- Higher weight = more important task
- Typical e-commerce: CTR=1.0, CTCVR=10-20× (conversion more valuable)
- Tune based on business metrics, not just ranking quality

## When to Use Explainability

```
┌─────────────────────────────────────────────────────────┐
│ Do you need to debug or understand fusion results?     │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ YES                           │ NO
         │                              │
         ▼                              ▼
┌──────────────────┐         ┌──────────────────────────┐
│ Use *_explain   │         │ Use regular functions     │
│ functions       │         │                          │
│                  │         │ • Faster                 │
│ • Debug why     │         │ • Less memory            │
│   documents     │         │                          │
│   ranked where  │         │                          │
│ • Identify      │         │                          │
│   retriever     │         │                          │
│   disagreements │         │                          │
│ • Build user-   │         │                          │
│   facing        │         │                          │
│   explanations  │         │                          │
│ • Validate      │         │                          │
│   fusion logic  │         │                          │
└──────────────────┘         └──────────────────────────┘
```

**Use Cases:**
- **Debugging**: "Why did this relevant doc rank so low?"
- **Validation**: "Are all retrievers contributing?"
- **User-facing**: "This result appears because it matched in both BM25 and semantic search"
- **Legal/Compliance**: "Show why this result was selected"

## Quick Reference Table

| Scenario | Algorithm | Config | Notes |
|----------|-----------|--------|-------|
| Unknown score scales | RRF | k=60 | Default for hybrid search |
| Same embedding model | CombSUM | Default | Fast, simple |
| Want to reward overlap | CombMNZ | Default | Amplifies consensus |
| Different distributions | DBSF or Standardized | Auto | Z-score normalization |
| Domain knowledge | Weighted | Custom weights | Tune based on evaluation |
| E-commerce multi-task | Additive Multi-Task | Task weights | CTR + CTCVR |
| Need debugging | *_explain variants | Same as base | Adds provenance |
| Very short lists (<10) | Score-based | - | Rank differences less meaningful |
| Highly correlated retrievers | Any | - | Fusion adds little value |

## Common Mistakes

1. **Using CombSUM with incompatible scales**
   - Problem: BM25 (0-100) + dense (0-1) without normalization
   - Fix: Use RRF or normalize properly

2. **Setting k too low in RRF**
   - Problem: k=1 overweights top positions, loses consensus signal
   - Fix: Use k=60 default, or k=20-40 if top retrievers are highly reliable

3. **Using CombMNZ with poor normalization**
   - Problem: CombMNZ amplifies normalization errors
   - Fix: Use CombSUM or ensure reliable normalization

4. **Not validating fusion results**
   - Problem: Silent failures (empty results, NaN scores)
   - Fix: Use `validate()` function after fusion

5. **Ignoring explainability in production**
   - Problem: Can't debug why results are wrong
   - Fix: Use `*_explain` functions, log explanations

