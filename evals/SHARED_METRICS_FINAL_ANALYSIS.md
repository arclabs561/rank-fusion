# Final Analysis: Shared `rank-metrics` Crate

## Executive Summary

**Recommendation: YES, create `rank-metrics` crate**

After re-evaluating with fresh perspective, the benefits outweigh the costs, especially for **TREC parsing** which is high value and low complexity.

## What Changed Since Last Analysis

### Previous Conclusion: "Don't Share"
- No immediate use case in other projects
- Type incompatibilities
- Different formulas (binary vs graded)
- Low maintenance burden

### New Perspective: "Share, But Strategically"

1. **TREC Parsing is Clearly Shareable**:
   - Standard format, well-tested
   - `rank-refine` could evaluate on TREC datasets (mentioned in plans)
   - Self-contained, low coupling
   - **High value, low complexity**

2. **Metrics Can Be Useful**:
   - `rank-refine` mentions nDCG improvements (evaluation context)
   - Having standard implementations helps consistency
   - Can be extended (e.g., differentiable for `rank-relax`)

3. **Workspace Structure**:
   - Separate workspaces (not unified)
   - Can use path dependencies for development
   - Can publish to crates.io later

## What to Share

### ✅ **Definitely Share**

#### 1. TREC Format Parsing
**Value:** ⭐⭐⭐⭐⭐ (Very High)
**Complexity:** ⭐ (Low)
**Coupling:** Low

**Extract:**
- `TrecRun` struct
- `Qrel` struct  
- `load_trec_runs()` function
- `load_qrels()` function
- `group_runs_by_query()` utility
- `group_qrels_by_query()` utility

**Why:**
- Standard format in IR research
- Well-tested, robust implementation
- Useful for `rank-refine` evaluation
- Self-contained

#### 2. Binary Relevance Metrics
**Value:** ⭐⭐⭐⭐ (High)
**Complexity:** ⭐ (Low)
**Coupling:** Low

**Extract:**
- `precision_at_k()`
- `recall_at_k()`
- `mrr()`
- `dcg_at_k()`
- `idcg_at_k()`
- `ndcg_at_k()`
- `average_precision()`
- `Metrics` struct (optional)

**Why:**
- Generic, pure functions
- Well-tested
- Useful for binary relevance scenarios
- Can be extended

### ⚠️ **Consider Sharing**

#### 3. Graded Relevance Metrics
**Value:** ⭐⭐⭐ (Medium)
**Complexity:** ⭐⭐ (Medium)
**Coupling:** Medium

**Extract:**
- `compute_ndcg()` (graded version)
- `compute_map()` (graded version)
- Helper functions

**Why:**
- Needed for real-world datasets
- Could be useful for `rank-refine`
- More tightly coupled to evaluation

**Decision:** Include but in separate module (`graded.rs`)

## Proposed Crate Structure

```
rank-metrics/
├── Cargo.toml
├── README.md
├── LICENSE-MIT
├── LICENSE-APACHE
└── src/
    ├── lib.rs           # Public API, re-exports
    ├── trec.rs          # TREC format parsing
    ├── binary.rs         # Binary relevance metrics
    ├── graded.rs         # Graded relevance metrics
    └── traits.rs         # Trait definitions (future extensibility)
```

## API Design

### Module Structure

```rust
// rank-metrics/src/lib.rs
pub mod trec;
pub mod binary;
pub mod graded;

// Re-export commonly used items
pub use trec::{TrecRun, Qrel, load_trec_runs, load_qrels};
pub use binary::{precision_at_k, recall_at_k, mrr, ndcg_at_k, average_precision};
pub use graded::{compute_ndcg_graded, compute_map_graded};
```

### Usage Example

```rust
// In rank-fusion-evals
use rank_metrics::{TrecRun, Qrel, load_trec_runs, load_qrels};
use rank_metrics::binary::ndcg_at_k;
use rank_metrics::graded::compute_ndcg_graded;

// Load TREC data
let runs = load_trec_runs("runs.txt")?;
let qrels = load_qrels("qrels.txt")?;

// Compute binary metrics
let ndcg = ndcg_at_k(&ranked, &relevant, 10);

// Compute graded metrics
let ndcg_graded = compute_ndcg_graded(&ranked, &qrels_map, 10);
```

## Implementation Plan

### Phase 1: Create Crate (30 min)
1. Create `rank-metrics` directory
2. Initialize Cargo project
3. Set up basic structure

### Phase 2: Extract TREC Parsing (1 hour)
1. Copy `TrecRun`, `Qrel` structs
2. Copy `load_trec_runs()`, `load_qrels()`
3. Copy grouping utilities
4. Move and adapt tests
5. Update error handling

### Phase 3: Extract Binary Metrics (1 hour)
1. Copy functions from `metrics.rs`
2. Keep generic design
3. Move tests
4. Make `Metrics` struct optional (serde feature)

### Phase 4: Extract Graded Metrics (1 hour)
1. Copy `compute_ndcg()`, `compute_map()`
2. Adapt to be more generic
3. Add tests
4. Document differences

### Phase 5: Update rank-fusion-evals (1 hour)
1. Add dependency
2. Update imports
3. Remove extracted code
4. Run tests
5. Fix any issues

### Phase 6: Testing & Documentation (1 hour)
1. Verify all tests pass
2. Write README
3. Add examples
4. Document API

**Total Estimated Time:** ~5-6 hours

## Benefits vs Costs

### Benefits ✅
- **TREC parsing shared** - Standard format, well-tested
- **Consistent metrics** - Single implementation across projects
- **Reusability** - `rank-refine` can use for evaluation
- **Extensibility** - Can add differentiable metrics for `rank-relax`
- **Maintainability** - Fix bugs once, benefit everywhere
- **Documentation** - Centralized docs for IR metrics

### Costs ⚠️
- **Migration effort** - ~5-6 hours to extract and test
- **Version management** - Coordinate versions (minimal with path deps)
- **Breaking changes** - Need to update all dependents
- **Dependency** - One more crate to maintain

### Net Assessment
**Benefits > Costs** - Especially for TREC parsing which is high value.

## Comparison: Share vs Don't Share

### If We Share:
- ✅ Standardized TREC parsing
- ✅ Consistent metrics
- ✅ Easier to extend
- ✅ Better documentation
- ⚠️ More initial work
- ⚠️ Version coordination

### If We Don't Share:
- ✅ Less initial work
- ✅ No version coordination
- ❌ Potential duplication if `rank-refine` needs metrics
- ❌ Inconsistent implementations
- ❌ Harder to extend

## Recommendation

**CREATE `rank-metrics` crate** with:
1. ✅ TREC parsing (definitely)
2. ✅ Binary metrics (definitely)
3. ✅ Graded metrics (yes, in separate module)

**Rationale:**
- TREC parsing is clearly shareable (high value, low complexity)
- Binary metrics are generic and useful
- Graded metrics are useful for real-world evaluation
- All are well-tested and stable
- Can be extended later (e.g., differentiable versions)
- Benefits outweigh costs

## Next Steps

1. **Confirm decision** - Proceed with creating `rank-metrics`?
2. **Create crate** - Initialize the project
3. **Extract code** - Follow implementation plan
4. **Test thoroughly** - Ensure nothing breaks
5. **Document** - Write comprehensive README
6. **Consider publishing** - To crates.io if useful

## Questions for You

1. **Scope**: Include all three (TREC + binary + graded) or just TREC + binary?
2. **Location**: Create in `/Users/arc/Documents/dev/rank-metrics`?
3. **Publishing**: Publish to crates.io or keep local for now?
4. **Timeline**: Do this now or later?

## My Strong Recommendation

**Do it now, include all three:**
- TREC parsing is clearly valuable
- Binary metrics are generic and useful
- Graded metrics complete the picture
- All are stable and well-tested
- Can always refine later

The effort is reasonable (~5-6 hours) and the benefits are clear, especially for TREC parsing which is a standard format that `rank-refine` could definitely use.

