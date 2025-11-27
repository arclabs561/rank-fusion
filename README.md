# rank-fusion

Combine ranked lists from multiple retrievers. Zero dependencies.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

```
cargo add rank-fusion
```

## What This Is

Fusion algorithms for hybrid search:

| Scenario | Algorithm |
|----------|-----------|
| BM25 + dense embeddings | `rrf` (rank-based) |
| Multiple retrievers, different scales | `rrf_multi` |
| Same-scale scores | `combsum`, `combmnz` |
| Trust one retriever more | `weighted`, `rrf_weighted` |
| Different distributions | `dbsf` (z-score) |

**What this is NOT**: embedding generation, reranking, vector search. See [rank-refine](https://crates.io/crates/rank-refine) for scoring embeddings.

## Usage

```rust
use rank_fusion::rrf;

let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
let dense = vec![("d2", 0.9), ("d3", 0.8)];

let fused = rrf(&bm25, &dense);
// [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
```

## API

### Rank-based (ignores scores)

| Function | Formula | Use |
|----------|---------|-----|
| `rrf(a, b)` | 1/(k + rank) | Different scales |
| `isr(a, b)` | 1/√(k + rank) | Lower ranks matter more |
| `borda(a, b)` | N - rank | Simple voting |

### Score-based

| Function | Formula | Use |
|----------|---------|-----|
| `combsum(a, b)` | Σ scores | Same scale |
| `combmnz(a, b)` | sum × count | Reward overlap |
| `dbsf(a, b)` | z-score | Different distributions |
| `weighted(a, b, config)` | weighted sum | Custom weights |

### Multi-list

All functions have `*_multi` variants:

```rust
use rank_fusion::{rrf_multi, RrfConfig};

let lists = vec![&bm25[..], &dense[..], &sparse[..]];
let fused = rrf_multi(&lists, RrfConfig::default());
```

### Weighted RRF

```rust
use rank_fusion::rrf_weighted;

let weights = [1.0, 2.0, 0.5];  // per-retriever
let fused = rrf_weighted(&lists, &weights, config)?;
```

## Formulas

**RRF** (Cormack 2009):
$$\text{RRF}(d) = \sum_r \frac{1}{k + \text{rank}_r(d)}$$

Default k=60. Rank is 0-indexed.

**CombMNZ**:
$$\text{score}(d) = \text{count}(d) \times \sum_r s_r(d)$$

**DBSF** (z-score normalization):
$$s' = \frac{s - \mu}{\sigma}, \quad \text{clipped to } [-3, 3]$$

## Benchmarks

Apple M3 Max, `cargo bench`:

| Operation | Items | Time |
|-----------|-------|------|
| `rrf` | 100 | 13μs |
| `rrf` | 1000 | 159μs |
| `combsum` | 100 | 14μs |
| `combmnz` | 100 | 13μs |
| `borda` | 100 | 13μs |
| `rrf_multi` (5 lists) | 100 | 38μs |

## For Library Authors

If you're building a search engine or RAG pipeline:

**Depend on it:**
```toml
rank-fusion = "0.1"
```

**Or vendor it:**
- `src/lib.rs` is self-contained (~2000 lines)
- Zero dependencies
- All algorithms in one file

See [DESIGN.md](DESIGN.md) for algorithm details.

## See Also

- [rank-refine](https://crates.io/crates/rank-refine): score with embeddings (cosine, MaxSim)
- [DESIGN.md](DESIGN.md): algorithm details and edge cases

## License

MIT OR Apache-2.0
