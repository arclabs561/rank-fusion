# rank-fusion

Rank fusion for hybrid search.

[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)

```rust
use rank_fusion::{rrf, RrfConfig};

let bm25 = vec![("doc1", 12.5), ("doc2", 11.2)];
let dense = vec![("doc2", 0.92), ("doc3", 0.85)];
let fused = rrf(bm25, dense, RrfConfig::default());
```

## Algorithms

| Function | Method | Uses Scores |
|----------|--------|-------------|
| `rrf` | Reciprocal Rank Fusion | No |
| `combsum` | Sum of normalized scores | Yes |
| `combmnz` | Sum × overlap count | Yes |
| `borda` | Borda count | No |
| `weighted` | Weighted combination | Yes |

All have `*_multi` variants for 3+ lists.

## License

MIT OR Apache-2.0
