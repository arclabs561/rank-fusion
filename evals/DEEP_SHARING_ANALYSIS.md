# Deep Analysis: Sharing Code Across Ranking Workspaces

## Executive Summary

After thorough analysis, **the case for sharing is WEAKER than initially thought**. Here's why:

### Key Findings:
1. **Metrics are NOT duplicated** - There are TWO different implementations serving DIFFERENT purposes
2. **No actual use case in other projects** - rank-refine and rank-relax don't currently need IR metrics
3. **Type incompatibility** - metrics.rs uses `HashSet`, real_world.rs uses `HashMap<String, u32>` (relevance scores)
4. **Different formulas** - NDCG implementations differ (binary vs graded relevance)
5. **Tight coupling** - Metrics struct uses serde, which adds dependency
6. **Low maintenance burden** - metrics.rs is stable, rarely changes

## Detailed Analysis

### 1. **Two Different Metric Implementations (NOT Duplication)**

#### Implementation A: `metrics.rs` (Generic, Binary Relevance)
- **Type**: `HashSet<I>` for relevant documents
- **Use Case**: Synthetic scenarios, binary relevance (relevant/not relevant)
- **NDCG Formula**: Binary relevance (1 if relevant, 0 if not)
- **Used By**: `main.rs` (synthetic evaluation)
- **Dependencies**: Only `std::collections::HashSet`

**Example:**
```rust
pub fn ndcg_at_k<I: Eq + std::hash::Hash>(
    ranked: &[I],
    relevant: &HashSet<I>,  // Binary: in set = relevant
    k: usize,
) -> f64
```

#### Implementation B: `real_world.rs` (Graded Relevance)
- **Type**: `HashMap<String, u32>` for relevance scores (0, 1, 2, 3...)
- **Use Case**: Real-world datasets with graded relevance judgments
- **NDCG Formula**: Uses actual relevance scores (u32)
- **Used By**: `evaluate_fusion_method` (real-world evaluation)
- **Dependencies**: Uses `HashMap` from real_world module

**Example:**
```rust
fn compute_ndcg(
    ranked: &[(String, f32)],
    qrels: &HashMap<String, u32>,  // Graded: 0, 1, 2, 3...
    k: usize,
) -> f64 {
    // Uses relevance as f64 in DCG calculation
    dcg += (relevance as f64) / ((rank + 2) as f64).log2();
}
```

**Why Two Implementations?**
- **Different data formats**: Binary (HashSet) vs Graded (HashMap with scores)
- **Different use cases**: Synthetic (simple) vs Real-world (complex)
- **Different formulas**: Binary relevance vs graded relevance in DCG

### 2. **Actual Usage Patterns**

#### `metrics.rs` Usage:
- **Used in**: `main.rs` (synthetic scenarios)
- **Pattern**: `Metrics::compute(ranked, relevant)` - single call
- **Frequency**: Once per scenario evaluation
- **Dependencies**: None (pure functions)

#### `real_world.rs` compute functions:
- **Used in**: `evaluate_fusion_method` → `compute_metrics`
- **Pattern**: Called per query, aggregates across queries
- **Frequency**: High (once per query × number of queries)
- **Dependencies**: Uses `TrecRun` and `Qrel` types

### 3. **Type Incompatibility Issues**

**Problem**: Can't easily unify because:
- `metrics.rs`: Generic over `I: Eq + Hash`, uses `HashSet<I>`
- `real_world.rs`: Specific to `String` IDs, uses `HashMap<String, u32>` for graded relevance

**To share, would need**:
- Generic relevance type (trait?)
- Support both binary and graded relevance
- More complex API

### 4. **Dependency Analysis**

#### `metrics.rs`:
```rust
use std::collections::HashSet;
// That's it! Zero external dependencies
```

#### `Metrics` struct (in metrics.rs):
```rust
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metrics {
    // ... fields
}
```
- **Uses serde** - adds dependency if extracted
- **Only used in**: `main.rs` for synthetic evaluation
- **Not used in**: Real-world evaluation (uses `FusionMetrics` instead)

#### `real_world.rs`:
- Uses `anyhow::Result`
- Uses `serde::{Serialize, Deserialize}`
- Uses `TrecRun` and `Qrel` types
- Tightly coupled to evaluation infrastructure

### 5. **Actual Use Cases in Other Projects**

#### rank-refine:
- **Current**: No evaluation metrics, no TREC datasets
- **Examples**: Focus on reranking, ColBERT scoring
- **Would need metrics for**: Benchmarking reranking quality
- **Reality**: No current need, hypothetical future use

#### rank-relax:
- **Current**: Differentiable ranking operations
- **Mentions NDCG**: In README as example use case
- **Would need**: Differentiable version of NDCG (not standard IR metrics)
- **Reality**: Needs different implementation (gradient-enabled)

### 6. **Code Statistics**

- **metrics.rs**: ~175 lines
- **Files using metrics.rs**: 2 (main.rs, integration_tests.rs)
- **Files using real_world.rs compute functions**: 1 (real_world.rs itself)
- **Total evals/src**: ~5000+ lines
- **Metrics as % of codebase**: ~3.5%

### 7. **Maintenance History**

- **metrics.rs**: Stable, rarely changes
- **Last significant change**: Initial implementation
- **Breaking changes**: None in recent history
- **Maintenance burden**: Low

### 8. **Compatibility Concerns**

#### Rust Version:
- rank-fusion: `rust-version = "1.74"`
- rank-refine: `rust-version = "1.74"`
- rank-relax: No explicit version (defaults to edition)
- **Compatible**: ✅ Yes

#### Edition:
- All use `edition = "2021"`
- **Compatible**: ✅ Yes

#### Dependencies:
- **Problem**: `metrics.rs` uses `serde::Serialize` for `Metrics` struct
- **If extracted**: Would need serde as dependency
- **Impact**: Adds dependency to shared crate (minor)

### 9. **Actual Benefits vs Costs**

#### Benefits (if shared):
- ✅ Standardized metrics across projects
- ✅ Single source of truth
- ⚠️ But: Only if other projects actually use them
- ⚠️ But: Type incompatibility means can't fully unify

#### Costs (if shared):
- ⚠️ Adds dependency management overhead
- ⚠️ Version coordination across projects
- ⚠️ Breaking changes affect multiple projects
- ⚠️ Need to maintain compatibility
- ⚠️ More complex CI/CD
- ⚠️ Migration effort (refactor imports)

#### Net Benefit:
- **Low** - Other projects don't currently need metrics
- **Hypothetical** - Future use case, not current need
- **Complex** - Type differences make sharing harder

### 10. **Alternative: Keep Separate, Document Patterns**

Instead of sharing, could:
1. **Document the pattern** - How to implement IR metrics
2. **Share examples** - Not code, but patterns
3. **Wait for actual need** - Extract when rank-refine actually needs metrics
4. **Keep lightweight** - metrics.rs is already minimal

## Recommendations

### Option A: **Don't Share (Recommended)**
**Rationale:**
- No current use case in other projects
- Type incompatibility makes sharing complex
- Low maintenance burden (metrics are stable)
- Small codebase (~175 lines)
- Different implementations serve different purposes

**Action**: Keep as-is, document the pattern for future reference

### Option B: **Share Only If Needed**
**Rationale:**
- Wait until rank-refine or rank-relax actually need metrics
- Extract then, when there's a concrete use case
- Avoid premature abstraction

**Action**: Monitor other projects, extract when needed

### Option C: **Share TREC Parsing Only** (If Needed)
**Rationale:**
- TREC format is standard
- More likely to be needed by other projects
- Less type complexity than metrics
- Can be extracted cleanly

**Action**: Extract `load_trec_runs` and `load_qrels` if other projects need TREC format

## Conclusion

**Recommendation: DON'T SHARE (for now)**

**Reasons:**
1. ✅ No actual use case (hypothetical only)
2. ✅ Type incompatibility (HashSet vs HashMap with scores)
3. ✅ Different formulas (binary vs graded relevance)
4. ✅ Low maintenance burden (stable code)
5. ✅ Small codebase (not worth the overhead)
6. ✅ Different purposes (synthetic vs real-world)

**When to Revisit:**
- When rank-refine actually needs IR metrics for evaluation
- When rank-relax needs standard (non-differentiable) metrics
- When there's concrete evidence of duplication across projects

**Better Approach:**
- Document the pattern in a guide
- Share examples, not code
- Extract when there's actual need, not hypothetical

