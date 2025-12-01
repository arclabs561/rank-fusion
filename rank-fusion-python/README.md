# rank-fusion-python

Python bindings for [`rank-fusion`](../rank-fusion/README.md) — rank fusion algorithms for hybrid search.

## Installation

### From PyPI

```bash
pip install rank-fusion-python
```

### From source

```bash
# Using uv (recommended)
cd rank-fusion-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv

# Or using pip
pip install maturin
maturin develop --release
```

## Usage

```python
import rank_fusion

# Two-list fusion
bm25 = [("d1", 12.5), ("d2", 11.0)]
dense = [("d2", 0.9), ("d3", 0.8)]
fused = rank_fusion.rrf(bm25, dense, k=60)

# Multi-list fusion
lists = [
    [("d1", 10.0), ("d2", 9.0)],
    [("d2", 0.9), ("d3", 0.8)],
]
fused = rank_fusion.rrf_multi(lists, k=60)

# Configuration
config = rank_fusion.RrfConfigPy(k=100)
```

## API

- `rrf(results_a, results_b, k=60)` — RRF fusion for two lists
- `rrf_multi(lists, k=60)` — RRF fusion for multiple lists
- `RrfConfigPy(k)` — Configuration for RRF fusion

## See Also

- [Core crate documentation](../rank-fusion/README.md)
- [Integration guide](../rank-fusion/INTEGRATION.md)
