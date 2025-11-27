# rank-fusion

Combine ranked lists from multiple retrievers.

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

```
cargo add rank-fusion
```

## Usage

```rust
use rank_fusion::rrf;

let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
let dense = vec![("d2", 0.9), ("d3", 0.8)];

let fused = rrf(&bm25, &dense);
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
| `weighted(a, b, wa, wb)` | weighted sum | Custom weights |

### Multi-list

All functions have `*_multi` variants:

```rust
use rank_fusion::{rrf_multi, RrfConfig};

let lists = vec![bm25, dense, sparse];
let fused = rrf_multi(&lists, RrfConfig::default());
```

### Weighted RRF

```rust
use rank_fusion::rrf_weighted;

let weights = vec![1.0, 2.0, 0.5];  // per-retriever
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

Apple M3, `cargo bench`:

| Operation | Items | Time |
|-----------|-------|------|
| `rrf` | 100 | 13μs |
| `rrf` | 1000 | 159μs |
| `combsum` | 100 | 14μs |
| `combmnz` | 100 | 13μs |
| `borda` | 100 | 13μs |
| `rrf_multi` (5 lists) | 100 | 38μs |

## See Also

- [rank-refine](https://crates.io/crates/rank-refine): rerank with embeddings
- [DESIGN.md](DESIGN.md): algorithm details

## License

MIT OR Apache-2.0
