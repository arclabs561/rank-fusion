# Proposal: Shared `rank-metrics` Crate

## Executive Summary

**Recommendation: CREATE a shared `rank-metrics` crate** with:
1. **TREC Format Parsing** (high value, low complexity)
2. **IR Evaluation Metrics** (binary and graded relevance)
3. **Extensible design** for future needs (e.g., differentiable metrics)

## Why Share Now?

### Changed Context Since Last Analysis

1. **TREC Parsing is High Value**:
   - Standard format used across IR research
   - `rank-refine` could evaluate on TREC datasets (mentioned in IMPLEMENTATION_PLANS.md)
   - Well-tested, robust implementation
   - Self-contained, low coupling

2. **Metrics Could Be Useful**:
   - `rank-refine` mentions nDCG improvements (evaluation context)
   - `rank-relax` mentions NDCG (though needs differentiable version)
   - Having a standard implementation helps consistency

3. **Workspace Benefits**:
   - Single source of truth for TREC format
   - Consistent metric implementations
   - Easier to extend (e.g., differentiable versions)

## Proposed Structure

```
rank-metrics/
├── Cargo.toml
├── README.md
└── src/
    ├── lib.rs
    ├── trec.rs          # TREC format parsing
    ├── metrics.rs       # Binary relevance metrics
    ├── graded.rs        # Graded relevance metrics
    └── traits.rs        # Trait definitions for extensibility
```

## What to Extract

### 1. TREC Format Parsing ✅ HIGH PRIORITY

**From:** `rank-fusion/evals/src/real_world.rs`

**Extract:**
- `TrecRun` struct
- `Qrel` struct
- `load_trec_runs()` function
- `load_qrels()` function
- `group_runs_by_query()` function (utility)
- `group_qrels_by_query()` function (utility)

**Dependencies:**
- `anyhow` (for error handling)
- `std::collections::HashMap`
- `std::fs::File`, `std::io::BufRead`

**Why Share:**
- Standard format, well-tested
- Useful for `rank-refine` if evaluating on TREC datasets
- Self-contained, low coupling

### 2. Binary Relevance Metrics ✅ MEDIUM PRIORITY

**From:** `rank-fusion/evals/src/metrics.rs`

**Extract:**
- `precision_at_k()`
- `recall_at_k()`
- `mrr()`
- `dcg_at_k()`
- `idcg_at_k()`
- `ndcg_at_k()`
- `average_precision()`
- `Metrics` struct (optional, or keep in rank-fusion)

**Dependencies:**
- `std::collections::HashSet`
- `serde` (for `Metrics` struct, if included)

**Why Share:**
- Generic, pure functions
- Useful for binary relevance scenarios
- Well-tested

### 3. Graded Relevance Metrics ⚠️ CONSIDER

**From:** `rank-fusion/evals/src/real_world.rs` (compute functions)

**Extract:**
- `compute_ndcg()` (graded version)
- `compute_map()` (graded version)
- Helper functions for graded relevance

**Dependencies:**
- `std::collections::HashMap`
- Uses `HashMap<String, u32>` for relevance scores

**Why Share:**
- Needed for real-world datasets with graded relevance
- Could be useful for `rank-refine` evaluation

**Consideration:**
- More tightly coupled to evaluation infrastructure
- Might keep in `rank-fusion-evals` for now

## Implementation Plan

### Phase 1: Create `rank-metrics` Crate (TREC + Binary Metrics)

1. **Create crate structure**
   ```bash
   cd /Users/arc/Documents/dev
   mkdir rank-metrics
   cd rank-metrics
   cargo init --lib
   ```

2. **Extract TREC parsing**
   - Move `TrecRun`, `Qrel` structs
   - Move `load_trec_runs()`, `load_qrels()`
   - Move grouping utilities
   - Add comprehensive tests

3. **Extract binary metrics**
   - Move functions from `metrics.rs`
   - Keep generic design
   - Add tests

4. **Update `rank-fusion-evals`**
   - Replace with `rank-metrics` dependency
   - Update imports
   - Verify tests pass

### Phase 2: Add Graded Metrics (Optional)

5. **Extract graded metrics** (if needed)
   - Move `compute_ndcg()`, `compute_map()` for graded relevance
   - Or keep in `rank-fusion-evals` if too tightly coupled

### Phase 3: Workspace Integration

6. **Add to workspace**
   - Update workspace `Cargo.toml`
   - Add `rank-metrics` as dependency in `rank-fusion-evals`
   - Update `rank-refine` to use if needed

## Benefits

### Immediate Benefits
- ✅ **TREC parsing shared** - Standard format, well-tested
- ✅ **Consistent metrics** - Single implementation
- ✅ **Easier testing** - Shared test suite

### Future Benefits
- ✅ **Extensibility** - Can add differentiable metrics for `rank-relax`
- ✅ **Consistency** - All projects use same metric implementations
- ✅ **Maintainability** - Fix bugs once, benefit everywhere

## Costs

### Initial Costs
- ⚠️ **Migration effort** - Extract code, update imports
- ⚠️ **Testing** - Ensure all tests pass after extraction
- ⚠️ **Documentation** - Document the shared crate

### Ongoing Costs
- ⚠️ **Version management** - Coordinate versions across projects
- ⚠️ **Breaking changes** - Need to coordinate across projects
- ⚠️ **Dependency** - Adds one more crate to maintain

## Recommendation

**CREATE `rank-metrics` crate with:**
1. ✅ **TREC parsing** (definitely share)
2. ✅ **Binary relevance metrics** (share)
3. ⚠️ **Graded relevance metrics** (consider, but may keep in evals)

**Rationale:**
- TREC parsing is high value, low complexity
- Binary metrics are generic and useful
- Graded metrics might stay in evals (more tightly coupled)
- Can always add more later

## Alternative: Minimal Approach

If we want to start smaller:

**Option A: Just TREC Parsing**
- Extract only TREC parsing to `rank-trec` crate
- Keep metrics in `rank-fusion-evals`
- Extract metrics later if needed

**Option B: Full Extraction**
- Extract everything (TREC + both metric types)
- More work upfront, but complete solution

## Decision Needed

**Question:** Should we:
1. Create `rank-metrics` with TREC + binary metrics? (Recommended)
2. Create `rank-trec` with just TREC parsing? (Minimal)
3. Extract everything including graded metrics? (Complete)

**My Recommendation:** Option 1 - Start with TREC + binary metrics, keep graded metrics in evals for now.

