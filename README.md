# rank-fusion

[![CI](https://github.com/arclabs561/rank-fusion/actions/workflows/ci.yml/badge.svg)](https://github.com/arclabs561/rank-fusion/actions)
[![Crates.io](https://img.shields.io/crates/v/rank-fusion.svg)](https://crates.io/crates/rank-fusion)
[![Docs](https://docs.rs/rank-fusion/badge.svg)](https://docs.rs/rank-fusion)
[![License](https://img.shields.io/badge/license-MIT%2FApache--2.0-blue.svg)](LICENSE)

Rank fusion algorithms for hybrid search — RRF, ISR, CombMNZ, Borda, DBSF, and more.

## Why Rank Fusion?

Hybrid search combines multiple retrievers (BM25, dense embeddings, sparse vectors) to get the best of each. **Problem**: Different retrievers use incompatible score scales. BM25 might score 0-100, while dense embeddings score 0-1. Normalization is fragile and requires tuning.

**Solution**: RRF (Reciprocal Rank Fusion) ignores scores and uses only rank positions. No normalization needed, works with any score distribution.

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

**Install from PyPI:**

```bash
pip install rank-fusion
```

```python
import rank_fusion

bm25 = [("d1", 12.5), ("d2", 11.0)]
dense = [("d2", 0.9), ("d3", 0.8)]
fused = rank_fusion.rrf(bm25, dense, k=60)
# [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
```

**For development/contributing:**

```bash
cd rank-fusion-python
uv venv
source .venv/bin/activate
uv tool install maturin
maturin develop --uv
```

### Node.js / WebAssembly

**Install from npm:**

```bash
npm install @arclabs561/rank-fusion
```

**Usage in Node.js:**

```javascript
const rankFusion = require('@arclabs561/rank-fusion');

// wasm-pack with --target nodejs exports functions directly
// Functions are available immediately after require()
const { rrf } = rankFusion;

// Fuse two ranked lists
const bm25 = [["d1", 12.5], ["d2", 11.0]];
const dense = [["d2", 0.9], ["d3", 0.8]];

// rrf returns Result<JsValue, JsValue> - unwrap if needed
const result = rrf(bm25, dense, 60);
const fused = result.Ok || result; // Handle Result type
// Result: [["d2", 0.033], ["d1", 0.016], ["d3", 0.016]]
// d2 ranks highest (appears in both lists)
```

**Usage in TypeScript:**

```typescript
import { rrf } from '@arclabs561/rank-fusion';

const bm25: [string, number][] = [["d1", 12.5], ["d2", 11.0]];
const dense: [string, number][] = [["d2", 0.9], ["d3", 0.8]];

const fused = rrf(bm25, dense, 60);
```

**Usage in Browser (ES Modules):**

```javascript
import init, { rrf } from '@arclabs561/rank-fusion';

async function fuseResults() {
  // Initialize WASM module
  await init();
  
  const bm25 = [["d1", 12.5], ["d2", 11.0]];
  const dense = [["d2", 0.9], ["d3", 0.8]];
  
  const fused = rrf(bm25, dense, 60);
  console.log("Fused results:", fused);
}

fuseResults();
```

## Documentation

- **[Core crate documentation](rank-fusion/README.md)** - Complete API reference and examples
- **[Python bindings](rank-fusion-python/README.md)** - Python usage guide
- **[Getting Started](rank-fusion/GETTING_STARTED.md)** - Tutorial with real-world examples
- **[Integration guide](rank-fusion/INTEGRATION.md)** - Framework-specific examples
- **[Design principles](rank-fusion/DESIGN.md)** - Algorithm details and theory
- **[Performance guide](rank-fusion/PERFORMANCE.md)** - Benchmarks and optimization tips
- **[API Documentation](https://docs.rs/rank-fusion)** - Full API reference on docs.rs
- **[Examples](rank-fusion/examples/)** - Runnable example code
- **[CHANGELOG](rank-fusion/CHANGELOG.md)** - Version history and changes

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


