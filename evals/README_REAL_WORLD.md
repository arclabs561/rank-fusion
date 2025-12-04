# Real-World Dataset Evaluation

This directory contains infrastructure for evaluating rank fusion methods on real-world IR datasets like MS MARCO, BEIR, and TREC.

## Quick Start

### 1. Prepare Your Dataset

Organize your dataset in the following structure:

```
datasets/
  msmarco/
    run1.txt          # TREC format run file
    run2.txt          # Another run file
    qrels.txt         # TREC format qrels
  beir-nq/
    run1.txt
    qrels.txt
  trec-dl-2023/
    run1.txt
    run2.txt
    qrels.txt
```

### 2. Run Evaluation

```bash
# Evaluate all datasets in a directory
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets

# Specify output files
cargo run --bin evaluate-real-world -- \
  --datasets-dir ./datasets \
  --output results.html \
  --json-output results.json
```

### 3. View Results

The evaluation generates:
- **HTML Report**: Interactive report with tables and visualizations
- **JSON Results**: Machine-readable results for further analysis

## Dataset Format

### TREC Run Format

Each line in a run file should be:
```
query_id Q0 doc_id rank score run_tag
```

Example:
```
1 Q0 doc123 1 0.95 bm25_run
1 Q0 doc456 2 0.87 bm25_run
2 Q0 doc789 1 0.92 bm25_run
```

### TREC Qrels Format

Each line in a qrels file should be:
```
query_id 0 doc_id relevance
```

Example:
```
1 0 doc123 2
1 0 doc456 1
2 0 doc789 2
```

Relevance levels:
- `0` = not relevant
- `1` = relevant
- `2+` = highly relevant (higher = more relevant)

## Supported Datasets

The evaluation framework supports any dataset in TREC format. Common datasets include:

### MS MARCO
- **Passage Ranking**: Large-scale passage retrieval
- **Document Ranking**: Document-level retrieval
- Available from: https://microsoft.github.io/msmarco/

### BEIR
- **18 datasets** across 9 domains
- Zero-shot evaluation benchmark
- Available from: https://github.com/beir-cellar/beir

### TREC Deep Learning Track
- **2023 Track**: 200 queries, 50+ runs
- Community-validated runs
- Available from: https://trec.nist.gov/

## Evaluation Methods

All fusion methods are evaluated:

- **RRF** (k=60): Reciprocal Rank Fusion
- **ISR** (k=1): Inverse Square Root
- **CombSUM**: Sum of normalized scores
- **CombMNZ**: Sum × overlap count
- **Borda**: Borda count voting
- **DBSF**: Distribution-Based Score Fusion
- **Weighted** (0.7/0.3, 0.9/0.1): Weighted combination
- **Standardized** (-3/3, -2/2): Z-score normalization
- **Additive Multi-Task** (1/1, 1/20): ResFlow-style fusion

## Metrics Reported

For each method and dataset:
- **nDCG@10**: Normalized Discounted Cumulative Gain at 10
- **nDCG@100**: Normalized Discounted Cumulative Gain at 100
- **MAP**: Mean Average Precision
- **MRR**: Mean Reciprocal Rank
- **P@10**: Precision at 10
- **R@100**: Recall at 100

## Programmatic Usage

You can also use the evaluation functions programmatically:

```rust
use rank_fusion_evals::real_world::*;
use rank_fusion_evals::dataset_loaders::*;

// Load runs and qrels
let runs = load_trec_runs("path/to/runs.txt")?;
let qrels = load_qrels("path/to/qrels.txt")?;

// Group by query
let grouped_runs = group_runs_by_query(&runs);
let grouped_qrels = group_qrels_by_query(&qrels);

// Evaluate all methods
let results = evaluate_all_methods(&grouped_runs, &grouped_qrels);

// Or evaluate a specific method
let method = FusionMethod::Standardized { clip_range: (-3.0, 3.0) };
let metrics = evaluate_fusion_method(&grouped_runs, &grouped_qrels, &method);
```

## Example: MS MARCO Evaluation

1. Download MS MARCO runs and qrels
2. Organize in `datasets/msmarco/`
3. Run evaluation:
   ```bash
   cargo run --bin evaluate-real-world -- --datasets-dir ./datasets
   ```

## Example: BEIR Evaluation

1. Download BEIR dataset (e.g., Natural Questions)
2. Generate runs using BM25 and dense retrieval
3. Organize in `datasets/beir-nq/`
4. Run evaluation

## Troubleshooting

### "No run files found"
- Ensure run files have `.run` or `.txt` extension
- Check that files are in the dataset directory

### "Empty runs or qrels"
- Verify run files contain valid TREC format data
- Check that qrels file exists and is readable

### "Failed to load runs"
- Verify TREC format is correct
- Check file permissions
- Ensure query IDs match between runs and qrels

## Next Steps

See `DATASET_RECOMMENDATIONS.md` for detailed recommendations on which datasets to use and how to obtain them.

