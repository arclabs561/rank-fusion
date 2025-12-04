# Sharing Code Across Ranking Workspaces

## Current Workspace Structure

- **rank-fusion**: Rank fusion algorithms (RRF, ISR, CombSUM, etc.) + evaluation infrastructure
- **rank-refine**: SIMD-accelerated similarity scoring (ColBERT, MaxSim) + reranking
- **rank-relax**: Differentiable ranking operations for ML training (candle/burn)

## Analysis: What Could Be Shared?

### ✅ **Good Candidates for Sharing**

#### 1. **Evaluation Metrics** (High Value)
- **Location**: `rank-fusion/evals/src/metrics.rs`
- **What**: NDCG, MAP, MRR, Precision@K, Recall@K, Average Precision
- **Why Share**: 
  - All three projects need evaluation metrics
  - `rank-relax` mentions NDCG in README (differentiable version)
  - `rank-refine` could benefit from standard IR metrics
  - Standardized implementation across projects
- **Current State**: Well-tested, pure functions, no dependencies
- **Sharing Method**: Extract to shared crate or publish as `rank-metrics`

#### 2. **TREC Format Parsing** (Medium Value)
- **Location**: `rank-fusion/evals/src/real_world.rs` (load_trec_runs, load_qrels)
- **What**: TREC run file and qrels parsing
- **Why Share**:
  - Standard format in IR research
  - `rank-refine` might evaluate on TREC datasets
  - `rank-relax` could use for training data
- **Current State**: Robust, handles edge cases
- **Sharing Method**: Extract to `rank-trec` crate

#### 3. **Dataset Statistics** (Low-Medium Value)
- **Location**: `rank-fusion/evals/src/dataset_statistics.rs`
- **What**: Comprehensive dataset analysis (score distributions, overlap, quality metrics)
- **Why Share**:
  - Useful for any ranking project evaluating on datasets
  - Helps understand dataset characteristics
- **Current State**: Well-structured, comprehensive
- **Sharing Method**: Part of shared evaluation crate

### ⚠️ **Maybe Share (Context-Dependent)**

#### 4. **Dataset Loaders** (Context-Specific)
- **Location**: `rank-fusion/evals/src/dataset_loaders.rs`
- **What**: MS MARCO, BEIR, MIRACL dataset loading
- **Why Maybe**:
  - `rank-refine` might use same datasets for evaluation
  - But: Different projects might need different datasets
  - But: Heavy dependencies (reqwest, tar, zip)
- **Recommendation**: Share only if both projects use same datasets

#### 5. **Dataset Converters** (Context-Specific)
- **Location**: `rank-fusion/evals/src/dataset_converters.rs`
- **What**: HuggingFace → TREC conversion
- **Why Maybe**:
  - Useful if `rank-refine` needs TREC format
  - But: Might have different conversion needs
- **Recommendation**: Share only if needed

### ❌ **Don't Share (Project-Specific)**

#### 6. **Fusion-Specific Evaluation**
- **Location**: `rank-fusion/evals/src/real_world.rs` (evaluate_fusion_method, etc.)
- **Why Not**: Very specific to fusion algorithms
- **Keep**: In rank-fusion workspace

#### 7. **Dataset Registry**
- **Location**: `rank-fusion/evals/src/dataset_registry.rs`
- **Why Not**: Curated for fusion evaluation needs
- **Keep**: In rank-fusion workspace

## Recommended Sharing Strategy

### Option 1: **Shared Metrics Crate** (Recommended)

Create a new crate: `rank-metrics` (or `rank-eval-core`)

**Structure:**
```
rank-metrics/
  src/
    lib.rs
    metrics.rs      # NDCG, MAP, MRR, Precision, Recall
    trec.rs         # TREC format parsing
    stats.rs        # Dataset statistics (optional)
  Cargo.toml
```

**Benefits:**
- ✅ Lightweight (no heavy dependencies)
- ✅ Reusable across all three projects
- ✅ Can be published to crates.io
- ✅ Easy to version independently

**Usage:**
```toml
# In rank-fusion/evals/Cargo.toml
[dependencies]
rank-metrics = { path = "../../rank-metrics" }

# In rank-refine/Cargo.toml (if needed)
[dependencies]
rank-metrics = { path = "../rank-metrics" }
```

### Option 2: **Monorepo with Shared Crate**

Create a parent workspace:
```
ranking-workspace/
  rank-fusion/
  rank-refine/
  rank-relax/
  rank-metrics/      # Shared crate
  Cargo.toml         # Workspace root
```

**Benefits:**
- ✅ Single workspace for all projects
- ✅ Easy path dependencies
- ✅ Shared CI/CD

**Drawbacks:**
- ⚠️ Requires restructuring existing repos
- ⚠️ Might complicate publishing

### Option 3: **Publish Shared Crate to crates.io**

Publish `rank-metrics` as a separate crate.

**Benefits:**
- ✅ No workspace restructuring needed
- ✅ Versioned independently
- ✅ Available to community
- ✅ Can be used by other projects

**Drawbacks:**
- ⚠️ Requires publishing workflow
- ⚠️ Version management across projects

## Recommendation

**Start with Option 1 (Shared Metrics Crate)**:

1. **Extract metrics to `rank-metrics` crate**:
   - NDCG, MAP, MRR, Precision@K, Recall@K, Average Precision
   - Keep it lightweight (no heavy deps)

2. **Extract TREC parsing** (if needed):
   - `load_trec_runs`, `load_qrels`
   - Basic TREC format support

3. **Use path dependencies initially**:
   ```toml
   rank-metrics = { path = "../rank-metrics" }
   ```

4. **Publish later if useful**:
   - Once stable, publish to crates.io
   - Other projects can use published version

## What NOT to Share (Yet)

- **Dataset loaders**: Too specific, heavy dependencies
- **Fusion evaluation**: Very specific to rank-fusion
- **Dataset registry**: Curated for fusion needs
- **Comprehensive statistics**: Might be overkill for other projects

## Next Steps

1. ✅ Research complete (this document)
2. ⚠️ Decide: Extract metrics crate now or later?
3. ⚠️ If yes: Create `rank-metrics` crate structure
4. ⚠️ Extract metrics.rs to shared crate
5. ⚠️ Update rank-fusion/evals to use shared crate
6. ⚠️ Test that everything still works

## Conclusion

**Yes, it makes sense to share metrics** - they're pure functions, well-tested, and useful across all ranking projects. 

**Maybe share TREC parsing** - if other projects need it.

**Don't share dataset loaders/registry** - too specific and heavy.

**Recommendation**: Start small with metrics, expand if needed.

