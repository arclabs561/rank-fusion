# Quick Start: Real-World Dataset Evaluation

## Installation

Everything is already set up! Just make sure dependencies are installed:

```bash
cargo build -p rank-fusion-evals
```

## Basic Usage

### 1. Prepare Your Dataset

Create a dataset directory with TREC format files:

```bash
mkdir -p datasets/my-dataset
# Place your run files and qrels.txt in datasets/my-dataset/
```

**TREC Run Format** (each line):
```
query_id Q0 doc_id rank score run_tag
```

**TREC Qrels Format** (each line):
```
query_id 0 doc_id relevance
```

### 2. Run Evaluation

```bash
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets
```

### 3. View Results

- **HTML Report**: `real_world_eval_report.html` (interactive, visual)
- **JSON Results**: `real_world_eval_results.json` (machine-readable)

## Example: Complete Workflow

```bash
# 1. Setup dataset directory
./evals/scripts/setup_dataset.sh msmarco

# 2. Place your TREC format files:
#    - datasets/msmarco/run1.txt (BM25 runs)
#    - datasets/msmarco/run2.txt (Dense runs)
#    - datasets/msmarco/qrels.txt (Relevance judgments)

# 3. Run evaluation
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets

# 4. Open the HTML report
open real_world_eval_report.html
```

## What Gets Evaluated

All 12 fusion method configurations:
- RRF (k=60)
- ISR (k=1)
- CombSUM
- CombMNZ
- Borda
- DBSF
- Weighted (0.7/0.3, 0.9/0.1)
- Standardized (-3/3, -2/2)
- Additive Multi-Task (1/1, 1/20)

## Metrics Reported

For each method:
- nDCG@10, nDCG@100
- MAP (Mean Average Precision)
- MRR (Mean Reciprocal Rank)
- Precision@10
- Recall@100

## Next Steps

- See `DATASET_RECOMMENDATIONS.md` for which datasets to use
- See `README_REAL_WORLD.md` for detailed documentation
- See `IMPLEMENTATION_COMPLETE.md` for what was implemented

