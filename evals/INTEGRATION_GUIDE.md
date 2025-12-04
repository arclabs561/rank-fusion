# Integration Guide: Real-World Dataset Evaluation

## Overview

The real-world evaluation system is fully integrated and ready to use. This guide shows how all components work together.

**Note:** Evaluation metrics and TREC parsing have been extracted to the shared `rank-eval` crate (see `../../rank-eval/README.md`). The `evals` crate now depends on `rank-eval` for these components.

## Architecture

```
evals/
├── src/
│   ├── lib.rs                    # Library entry point (exports all modules)
│   ├── main.rs                   # Synthetic scenario evaluation
│   ├── real_world.rs             # Core evaluation functions (uses rank-eval)
│   ├── metrics.rs                # Binary metrics (re-exports from rank-eval)
│   ├── dataset_loaders.rs        # Dataset loading utilities
│   ├── evaluate_real_world.rs    # Comprehensive evaluation pipeline
│   └── bin/
│       └── evaluate_real_world.rs # CLI binary
├── DATASET_RECOMMENDATIONS.md    # Research-backed dataset guide
├── README_REAL_WORLD.md          # Usage documentation
├── QUICK_START.md                # Quick start guide
└── scripts/                      # Helper scripts

Dependencies:
└── rank-eval/                    # Shared evaluation crate
    ├── trec.rs                   # TREC format parsing
    ├── binary.rs                 # Binary relevance metrics
    └── graded.rs                 # Graded relevance metrics
```

## Component Integration

### 1. Library Module (`lib.rs`)

Exports all modules for use by binaries and external code:

```rust
pub mod datasets;           // Synthetic scenarios
pub mod metrics;            // IR metrics computation
pub mod real_world;         // Real-world evaluation core
pub mod dataset_loaders;    // Dataset loading
pub mod evaluate_real_world; // Evaluation pipeline
```

### 2. Real-World Evaluation Core (`real_world.rs`)

**Key Functions:**
- `load_trec_runs()` - Load TREC format run files
- `load_qrels()` - Load TREC format qrels
- `group_runs_by_query()` - Organize runs by query
- `evaluate_fusion_method()` - Evaluate single method
- `evaluate_all_methods()` - Evaluate all 12 methods
- `FusionMethod` enum - All fusion method configurations

**Supported Methods:**
- RRF (k=60)
- ISR (k=1)
- CombSUM, CombMNZ, Borda, DBSF
- Weighted (0.7/0.3, 0.9/0.1)
- Standardized (-3/3, -2/2)
- Additive Multi-Task (1/1, 1/20)

### 3. Dataset Loaders (`dataset_loaders.rs`)

**Functions:**
- `load_msmarco_runs()` - MS MARCO dataset loading
- `load_beir_runs()` - BEIR dataset loading
- `load_trec_runs_from_dir()` - Generic TREC loading
- `validate_dataset_dir()` - Dataset validation
- `get_dataset_stats()` - Compute statistics

### 4. Evaluation Pipeline (`evaluate_real_world.rs`)

**Key Functions:**
- `evaluate_dataset()` - Single dataset evaluation
- `evaluate_datasets_dir()` - Multi-dataset evaluation
- `generate_html_report()` - HTML report generation
- `compute_summary()` - Cross-dataset statistics

**Output:**
- `EvaluationResults` - Complete results structure
- HTML reports with visualizations
- JSON results for programmatic access

### 5. CLI Binary (`bin/evaluate_real_world.rs`)

**Usage:**
```bash
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets
```

**Features:**
- Progress reporting
- Summary statistics
- HTML and JSON output
- Error handling

## Data Flow

```
TREC Run Files + Qrels
    ↓
dataset_loaders::load_*()
    ↓
real_world::group_runs_by_query()
    ↓
evaluate_real_world::evaluate_dataset()
    ↓
real_world::evaluate_all_methods()
    ↓
Metrics Computation (nDCG, MAP, MRR, etc.)
    ↓
evaluate_real_world::generate_html_report()
    ↓
HTML Report + JSON Results
```

## Usage Patterns

### Pattern 1: Command-Line Evaluation

```bash
# Setup dataset
./evals/scripts/setup_dataset.sh msmarco

# Place TREC files in datasets/msmarco/
# - run1.txt (BM25 runs)
# - run2.txt (Dense runs)
# - qrels.txt (Relevance judgments)

# Run evaluation
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets

# View results
open real_world_eval_report.html
```

### Pattern 2: Programmatic Usage

```rust
use rank_fusion_evals::real_world::*;
use rank_fusion_evals::dataset_loaders::*;

// Load data
let runs = load_trec_runs("runs.txt")?;
let qrels = load_qrels("qrels.txt")?;

// Group by query
let grouped_runs = group_runs_by_query(&runs);
let grouped_qrels = group_qrels_by_query(&qrels);

// Evaluate all methods
let results = evaluate_all_methods(&grouped_runs, &grouped_qrels);

// Or evaluate specific method
let method = FusionMethod::Standardized { clip_range: (-3.0, 3.0) };
let metrics = evaluate_fusion_method(&grouped_runs, &grouped_qrels, &method);
```

### Pattern 3: Multi-Dataset Evaluation

```rust
use rank_fusion_evals::evaluate_real_world::*;

// Evaluate all datasets in directory
let results = evaluate_datasets_dir("./datasets")?;

// Generate report
generate_html_report(&results, "report.html")?;

// Access summary
println!("Datasets: {}", results.summary.total_datasets);
println!("Best method: {:?}", results.summary.best_methods_per_dataset);
```

## Metrics Computed

For each fusion method:

1. **nDCG@10** - Normalized Discounted Cumulative Gain at 10
2. **nDCG@100** - Normalized Discounted Cumulative Gain at 100
3. **MAP** - Mean Average Precision
4. **MRR** - Mean Reciprocal Rank
5. **Precision@10** - Precision at 10
6. **Recall@100** - Recall at 100

## Error Handling

All functions use `anyhow::Result` for comprehensive error handling:

```rust
use anyhow::{Context, Result};

let runs = load_trec_runs("runs.txt")
    .with_context(|| "Failed to load runs")?;
```

## Testing

Run tests with:

```bash
# All tests
cargo test -p rank-fusion-evals

# Specific module
cargo test -p rank-fusion-evals real_world

# With output
cargo test -p rank-fusion-evals -- --nocapture
```

## Performance Considerations

- **Efficient grouping**: Uses HashMap for O(1) lookups
- **Lazy evaluation**: Metrics computed only when needed
- **Memory efficient**: Processes queries one at a time
- **Parallelizable**: Can be extended for parallel query processing

## Extension Points

### Adding New Fusion Methods

1. Add to `FusionMethod` enum in `real_world.rs`
2. Implement `fuse()` method
3. Add to `evaluate_all_methods()` default list

### Adding New Metrics

1. Add field to `FusionMetrics` struct
2. Implement computation in `compute_metrics()`
3. Update HTML report generation

### Adding New Dataset Formats

1. Add loader function in `dataset_loaders.rs`
2. Convert to TREC format internally
3. Use existing TREC evaluation pipeline

## Troubleshooting

### "No run files found"
- Check file extensions (.run or .txt)
- Verify files are in dataset directory
- Check file permissions

### "Empty runs or qrels"
- Verify TREC format is correct
- Check for empty files
- Ensure query IDs match between runs and qrels

### "Failed to load runs"
- Verify TREC format syntax
- Check for encoding issues (should be UTF-8)
- Ensure proper whitespace separation

## Next Steps

1. **Download datasets**: Use recommendations in `DATASET_RECOMMENDATIONS.md`
2. **Run evaluation**: Use CLI or programmatic API
3. **Analyze results**: Review HTML reports and JSON data
4. **Compare methods**: Use summary statistics to identify best methods
5. **Iterate**: Adjust fusion parameters based on results

## Research Integration

The implementation is based on:
- **ERANK**: Standardized fusion (z-score normalization)
- **ResFlow**: Additive multi-task fusion
- **TREC**: Standard evaluation protocols
- **BEIR**: Zero-shot evaluation methodology

All methods are validated against research findings and can be compared with published results.

