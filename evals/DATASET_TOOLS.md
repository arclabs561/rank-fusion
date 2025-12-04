# Dataset Tools and Utilities

This document describes the tools and utilities available for working with evaluation datasets.

## Available Tools

### 1. Dataset Registry (`list-datasets` binary)

List and explore available datasets:

```bash
# List all datasets
cargo run --bin list-datasets

# List by priority
cargo run --bin list-datasets -- --priority 1

# List by category
cargo run --bin list-datasets -- --category multilingual

# Detailed view
cargo run --bin list-datasets -- --detailed

# JSON output
cargo run --bin list-datasets -- --json
```

**Categories:**
- `general` - General-purpose datasets
- `multilingual` - Multilingual datasets
- `domain` - Domain-specific datasets
- `qa` - Question answering datasets
- `regional` - Regional/language-specific
- `specialized` - Specialized use cases

### 2. Dataset Conversion (`convert_hf_to_trec.py`)

Convert HuggingFace datasets to TREC format:

```bash
# Convert MIRACL (English)
python evals/scripts/convert_hf_to_trec.py \
  --dataset mteb/miracl \
  --language en \
  --split dev \
  --output-dir ./datasets/miracl-en

# Convert LoTTE
python evals/scripts/convert_hf_to_trec.py \
  --dataset mteb/LoTTE \
  --split test \
  --output-dir ./datasets/lotte

# Convert HotpotQA
python evals/scripts/convert_hf_to_trec.py \
  --dataset hotpotqa/hotpotqa \
  --split validation \
  --output-dir ./datasets/hotpotqa
```

**Requirements:**
```bash
pip install datasets
```

### 3. Dataset Setup (`setup_all_datasets.sh`)

Create directory structure for all recommended datasets:

```bash
./evals/scripts/setup_all_datasets.sh ./datasets
```

This creates directories for all 21+ datasets organized by priority.

### 4. Dataset Registry Generation

Generate a JSON registry of all datasets:

```bash
./evals/scripts/generate_dataset_registry.sh ./datasets/registry.json
```

Or use the Rust binary directly:

```bash
cargo run --bin list-datasets -- --json > datasets/registry.json
```

## Dataset Loaders (Rust API)

### Load TREC Format Datasets

```rust
use rank_fusion_evals::dataset_loaders::*;

// Load TREC runs
let runs = load_trec_runs_from_dir("./datasets/msmarco", &["run1.txt", "run2.txt"])?;

// Load TREC qrels
let qrels = load_trec_qrels_from_dir("./datasets/msmarco")?;
```

### Load Specific Dataset Types

```rust
use rank_fusion_evals::dataset_loaders::*;

// MS MARCO
let runs = load_msmarco_runs("./datasets/msmarco", &["bm25.txt", "dense.txt"])?;
let qrels = load_msmarco_qrels("./datasets/msmarco/qrels.txt")?;

// BEIR
let runs = load_beir_runs("./datasets/beir-nq/runs.txt")?;
let qrels = load_beir_qrels("./datasets/beir-nq/qrels.txt")?;

// MIRACL (after conversion)
let runs = load_miracl_runs("./datasets/miracl-en/runs.txt")?;
let qrels = load_miracl_qrels("./datasets/miracl-en/qrels.txt")?;
```

### Dataset Registry (Rust API)

```rust
use rank_fusion_evals::dataset_registry::*;

let registry = DatasetRegistry::new();

// Get dataset by name
let msmarco = registry.get("msmarco-passage").unwrap();

// Get all priority 1 datasets
let essential = registry.by_priority(1);

// Get all multilingual datasets
let multilingual = registry.by_category(DatasetCategory::Multilingual);

// List all dataset names
let names = registry.list_names();
```

## Dataset Converters (Rust API)

### Convert HuggingFace Format

```rust
use rank_fusion_evals::dataset_converters::*;

// Convert JSONL to TREC runs
convert_jsonl_to_trec_runs(
    "input.jsonl",
    "output_runs.txt",
    "my_run_tag"
)?;

// Convert JSONL to TREC qrels
convert_jsonl_to_trec_qrels(
    "input_qrels.jsonl",
    "output_qrels.txt"
)?;
```

### Conversion Configuration

```rust
use rank_fusion_evals::dataset_converters::*;

let config = ConversionConfig {
    input_format: "jsonl".to_string(),
    output_format: "trec".to_string(),
    input_path: PathBuf::from("input.jsonl"),
    output_path: PathBuf::from("output.txt"),
    run_tag: Some("converted".to_string()),
};

convert_dataset(&config)?;
```

## Complete Workflow Example

### 1. Setup Dataset Directories

```bash
./evals/scripts/setup_all_datasets.sh ./datasets
```

### 2. Download and Convert HuggingFace Dataset

```bash
# Convert MIRACL English
python evals/scripts/convert_hf_to_trec.py \
  --dataset mteb/miracl \
  --language en \
  --output-dir ./datasets/miracl-en
```

### 3. Download TREC Format Dataset

```bash
# MS MARCO (already in TREC format)
# Download from https://microsoft.github.io/msmarco/
# Place runs and qrels in ./datasets/msmarco/
```

### 4. Validate Dataset

```rust
use rank_fusion_evals::dataset_loaders::*;

let is_valid = validate_dataset_dir("./datasets/msmarco")?;
if is_valid {
    println!("Dataset is valid!");
}
```

### 5. Get Dataset Statistics

```rust
use rank_fusion_evals::dataset_loaders::*;
use rank_fusion_evals::real_world::*;

let runs = load_trec_runs("./datasets/msmarco/runs.txt")?;
let qrels = load_qrels("./datasets/msmarco/qrels.txt")?;

let stats = get_dataset_stats(&runs, &qrels);
println!("Queries: {}", stats.unique_queries);
println!("Documents: {}", stats.unique_documents);
println!("Runs: {}", stats.unique_run_tags);
```

### 6. Run Evaluation

```bash
cargo run --bin evaluate-real-world -- --datasets-dir ./datasets
```

## Format Conversion Examples

### HuggingFace → TREC

Most HuggingFace datasets need conversion. The Python script handles common formats:

**MIRACL format:**
```json
{
  "query_id": "1",
  "query": "What is...",
  "positive_passages": [
    {"docid": "doc1", "title": "...", "text": "..."},
    {"docid": "doc2", "title": "...", "text": "..."}
  ]
}
```

**LoTTE format:**
```json
{
  "query_id": "1",
  "query": "...",
  "positive_passages": [...],
  "negative_passages": [...]
}
```

### BEIR → TREC

BEIR datasets can be loaded via Python and converted:

```python
from beir import util, LoggingHandler
from beir.datasets.data_loader import GenericDataLoader

# Download and load BEIR dataset
url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
out_dir = "./datasets/beir-nq"
data_path = util.download_and_unzip(url, out_dir)

corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")

# Convert to TREC format
# (Write conversion script based on BEIR structure)
```

## Dataset Validation

Before running evaluation, validate your dataset:

```rust
use rank_fusion_evals::dataset_loaders::*;

// Check if directory has required files
let is_valid = validate_dataset_dir("./datasets/my-dataset")?;

if !is_valid {
    println!("Dataset missing required files!");
    println!("Need: at least one .run or .txt file");
    println!("Need: qrels.txt or qrels file");
}
```

## Troubleshooting

### "No qrels file found"
- Check file names: `qrels.txt`, `qrels`, `qrels.dev.txt`, `qrels.test.txt`
- Ensure file is in dataset directory
- Check file permissions

### "Failed to parse JSON"
- Verify JSON format is correct
- Check for encoding issues (should be UTF-8)
- Validate JSON structure matches expected format

### "Dataset format not supported"
- Convert to TREC format first
- Use conversion scripts provided
- Check dataset documentation for format specifications

## Next Steps

1. **Explore datasets**: Use `list-datasets` to see available options
2. **Download datasets**: Follow instructions in `DATASET_RECOMMENDATIONS.md`
3. **Convert formats**: Use conversion scripts for HuggingFace datasets
4. **Validate**: Check dataset structure before evaluation
5. **Evaluate**: Run evaluation on prepared datasets

See `DATASET_RECOMMENDATIONS.md` for detailed information on each dataset.

