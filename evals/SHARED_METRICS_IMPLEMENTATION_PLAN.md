# Implementation Plan: Shared `rank-metrics` Crate

## Current Situation

- **rank-fusion**: Has TREC parsing + binary metrics + graded metrics
- **rank-refine**: No metrics/TREC parsing currently
- **rank-relax**: No metrics/TREC parsing (needs differentiable version)
- **Workspaces**: Separate (not unified)

## Decision: Create Shared Crate

**Recommendation: CREATE `rank-metrics` crate** as a separate project that all three workspaces can depend on.

## Implementation Strategy

### Option A: Path Dependencies (Development)
Each workspace depends on `rank-metrics` via path:
```toml
[dependencies]
rank-metrics = { path = "../../rank-metrics" }
```

### Option B: Published Crate (Production)
Publish to crates.io, each workspace depends on published version:
```toml
[dependencies]
rank-metrics = "0.1"
```

### Option C: Git Dependency (Alternative)
Use git URL if not publishing:
```toml
[dependencies]
rank-metrics = { git = "https://github.com/arclabs561/rank-metrics" }
```

**Recommendation:** Start with Option A (path), publish later if needed.

## What to Extract

### Phase 1: Core Extraction (High Priority)

#### 1. TREC Format Parsing ✅
**From:** `rank-fusion/evals/src/real_world.rs`

**Extract:**
```rust
// Types
pub struct TrecRun { ... }
pub struct Qrel { ... }

// Functions
pub fn load_trec_runs(path: impl AsRef<Path>) -> Result<Vec<TrecRun>>
pub fn load_qrels(path: impl AsRef<Path>) -> Result<Vec<Qrel>>
pub fn group_runs_by_query(runs: &[TrecRun]) -> HashMap<String, HashMap<String, Vec<(String, f32)>>>
pub fn group_qrels_by_query(qrels: &[Qrel]) -> HashMap<String, HashMap<String, u32>>
```

**Dependencies:**
- `anyhow` (error handling)
- `std` only

**Value:** ⭐⭐⭐⭐⭐ (High - standard format, well-tested)

#### 2. Binary Relevance Metrics ✅
**From:** `rank-fusion/evals/src/metrics.rs`

**Extract:**
```rust
// Functions (all generic)
pub fn precision_at_k<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64
pub fn recall_at_k<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64
pub fn mrr<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64
pub fn dcg_at_k<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64
pub fn idcg_at_k(n_relevant: usize, k: usize) -> f64
pub fn ndcg_at_k<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64
pub fn average_precision<I: Eq + Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64

// Optional: Metrics struct
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metrics { ... }
```

**Dependencies:**
- `std::collections::HashSet`
- `serde` (optional, for Metrics struct)

**Value:** ⭐⭐⭐⭐ (Medium-High - generic, useful)

### Phase 2: Graded Metrics (Consider)

#### 3. Graded Relevance Metrics ⚠️
**From:** `rank-fusion/evals/src/real_world.rs`

**Extract:**
```rust
// Functions for graded relevance (HashMap<String, u32>)
pub fn compute_ndcg_graded(ranked: &[(String, f32)], qrels: &HashMap<String, u32>, k: usize) -> f64
pub fn compute_map_graded(ranked: &[(String, f32)], qrels: &HashMap<String, u32>) -> f64
```

**Dependencies:**
- `std::collections::HashMap`
- More tightly coupled to evaluation infrastructure

**Value:** ⭐⭐⭐ (Medium - useful but more specific)

**Decision:** Extract to `rank-metrics` but in separate module (`graded.rs`)

## Proposed Crate Structure

```
rank-metrics/
├── Cargo.toml
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
└── src/
    ├── lib.rs           # Public API
    ├── trec.rs          # TREC format parsing
    ├── binary.rs         # Binary relevance metrics
    ├── graded.rs         # Graded relevance metrics
    └── traits.rs         # Trait definitions (for future extensibility)
```

## Cargo.toml

```toml
[package]
name = "rank-metrics"
version = "0.1.0"
edition = "2021"
rust-version = "1.74"
license = "MIT OR Apache-2.0"
authors = ["Arc <attobop@gmail.com>"]
description = "IR evaluation metrics and TREC format parsing for Rust"
repository = "https://github.com/arclabs561/rank-metrics"
documentation = "https://docs.rs/rank-metrics"
keywords = ["ir", "information-retrieval", "metrics", "ndcg", "trec", "evaluation"]
categories = ["algorithms", "science"]

[dependencies]
anyhow = "1.0"
serde = { version = "1.0", features = ["derive"], optional = true }

[features]
default = ["serde"]
```

## Migration Steps

### Step 1: Create `rank-metrics` Crate

```bash
cd /Users/arc/Documents/dev
cargo new --lib rank-metrics
cd rank-metrics
```

### Step 2: Extract TREC Parsing

1. Copy `TrecRun`, `Qrel` structs
2. Copy `load_trec_runs()`, `load_qrels()` functions
3. Copy grouping utilities
4. Add tests (move from integration_tests.rs)
5. Update error handling to use `anyhow`

### Step 3: Extract Binary Metrics

1. Copy all functions from `metrics.rs`
2. Keep generic design
3. Move tests
4. Make `Metrics` struct optional (behind `serde` feature)

### Step 4: Extract Graded Metrics (Optional)

1. Copy `compute_ndcg()`, `compute_map()` from `real_world.rs`
2. Adapt to be more generic if possible
3. Add tests

### Step 5: Update `rank-fusion-evals`

1. Add dependency:
   ```toml
   [dependencies]
   rank-metrics = { path = "../../rank-metrics" }
   ```

2. Update imports:
   ```rust
   use rank_metrics::{TrecRun, Qrel, load_trec_runs, load_qrels};
   use rank_metrics::binary::{ndcg_at_k, precision_at_k, ...};
   use rank_metrics::graded::{compute_ndcg, compute_map};
   ```

3. Remove extracted code
4. Run tests to verify

### Step 6: Test Integration

1. Verify `rank-fusion-evals` tests pass
2. Verify `rank-fusion-evals` binaries work
3. Check that nothing breaks

## Benefits Analysis

### Immediate Benefits
- ✅ **TREC parsing shared** - Standard format, well-tested
- ✅ **Consistent metrics** - Single implementation
- ✅ **Reusable** - `rank-refine` can use TREC parsing for evaluation

### Future Benefits
- ✅ **Extensibility** - Can add differentiable metrics for `rank-relax`
- ✅ **Consistency** - All projects use same implementations
- ✅ **Maintainability** - Fix bugs once, benefit everywhere

## Costs

### Initial Costs
- ⚠️ **Migration effort** - ~2-4 hours to extract and test
- ⚠️ **Testing** - Need to ensure all tests pass
- ⚠️ **Documentation** - Document the new crate

### Ongoing Costs
- ⚠️ **Version management** - Coordinate versions (minimal if using path deps)
- ⚠️ **Breaking changes** - Need to update all dependents
- ⚠️ **Dependency** - One more crate to maintain

## Recommendation

**CREATE `rank-metrics` crate with:**
1. ✅ **TREC parsing** (definitely)
2. ✅ **Binary metrics** (definitely)
3. ✅ **Graded metrics** (yes, in separate module)

**Rationale:**
- TREC parsing is high value, low complexity
- Binary metrics are generic and useful
- Graded metrics are useful for real-world evaluation
- All three are well-tested and stable
- Can be extended later (e.g., differentiable versions)

## Next Steps

1. **Get approval** on this plan
2. **Create `rank-metrics` crate**
3. **Extract code** following the plan
4. **Update `rank-fusion-evals`** to use the new crate
5. **Test thoroughly**
6. **Document the crate**
7. **Consider publishing** to crates.io if useful

## Questions to Answer

1. **Location**: Create in `/Users/arc/Documents/dev/rank-metrics`?
2. **Publishing**: Publish to crates.io or keep local?
3. **Scope**: Include graded metrics or just TREC + binary?
4. **Timeline**: Do this now or wait?

