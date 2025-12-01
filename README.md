# rank-fusion

Rank fusion algorithms for hybrid search — RRF, ISR, CombMNZ, Borda, DBSF, and more.

This repository contains a Cargo workspace with multiple crates:

- **[`rank-fusion`](rank-fusion/)** — Core library (zero dependencies by default)
- **[`rank-fusion-python`](rank-fusion-python/)** — Python bindings using PyO3

## Quick Start

### Rust

```bash
cargo add rank-fusion
```

```rust
use rank_fusion::rrf;

let bm25 = vec![("d1", 12.5), ("d2", 11.0)];
let dense = vec![("d2", 0.9), ("d3", 0.8)];
let fused = rrf(&bm25, &dense);
// d2 ranks highest (appears in both lists)
```

### Python

**Using uv (recommended):**

```bash
cd rank-fusion-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

**Or using pip:**

```bash
cd rank-fusion-python
pip install maturin
maturin develop --release
```

```python
import rank_fusion

bm25 = [("d1", 12.5), ("d2", 11.0)]
dense = [("d2", 0.9), ("d3", 0.8)]
fused = rank_fusion.rrf(bm25, dense, k=60)
```

## Documentation

- [Core crate documentation](rank-fusion/README.md)
- [Python bindings](rank-fusion-python/README.md)
- [Integration guide](rank-fusion/INTEGRATION.md)
- [Design principles](rank-fusion/DESIGN.md)

## Development

```bash
# Build core crate (fast, no Python required)
cargo build -p rank-fusion

# Build all workspace members
cargo build --workspace

# Test core crate
cargo test -p rank-fusion

# Test all workspace members
cargo test --workspace

# Check Python bindings (requires Python installed)
cargo check -p rank-fusion-python

# Build Python bindings with uv
cd rank-fusion-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

## Workspace Structure

This repository uses a Cargo workspace to organize the codebase:

- **Shared target directory** — All crates compile to one `target/` directory
- **Workspace inheritance** — Dependencies and versions defined once at workspace root
- **Path dependencies** — Python crate depends on core via path (no version conflicts)
- **Default members** — Only core crate builds by default (`cargo build`)

See [`archive/2025-01/LIGHTWEIGHT_WORKSPACE_PATTERNS.md`](archive/2025-01/LIGHTWEIGHT_WORKSPACE_PATTERNS.md) for details on the workspace design.

