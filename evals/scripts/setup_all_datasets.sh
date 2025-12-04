#!/bin/bash
# Setup script for creating dataset directory structure for all recommended datasets

set -e

DATASETS_DIR="${1:-./datasets}"
mkdir -p "$DATASETS_DIR"

echo "Creating dataset directory structure in: $DATASETS_DIR"
echo ""

# Priority 1: Essential
echo "Priority 1: Essential Datasets"
mkdir -p "$DATASETS_DIR/msmarco"
mkdir -p "$DATASETS_DIR/beir"

# Priority 2: High Value
echo "Priority 2: High Value"
mkdir -p "$DATASETS_DIR/trec-dl-2023"
mkdir -p "$DATASETS_DIR/lotte"

# Priority 3: Multilingual
echo "Priority 3: Multilingual"
mkdir -p "$DATASETS_DIR/miracl"
mkdir -p "$DATASETS_DIR/mteb"

# Priority 4: Domain-Specific
echo "Priority 4: Domain-Specific"
mkdir -p "$DATASETS_DIR/legalbench-rag"
mkdir -p "$DATASETS_DIR/fiqa"
mkdir -p "$DATASETS_DIR/bioasq"
mkdir -p "$DATASETS_DIR/scifact-open"

# Priority 5: Question Answering
echo "Priority 5: Question Answering"
mkdir -p "$DATASETS_DIR/hotpotqa"
mkdir -p "$DATASETS_DIR/natural-questions"
mkdir -p "$DATASETS_DIR/squad"

# Priority 6: Regional
echo "Priority 6: Regional"
mkdir -p "$DATASETS_DIR/fire"
mkdir -p "$DATASETS_DIR/clef"
mkdir -p "$DATASETS_DIR/ntcir"

# Priority 7: Specialized
echo "Priority 7: Specialized"
mkdir -p "$DATASETS_DIR/fultr"
mkdir -p "$DATASETS_DIR/trec-covid"
mkdir -p "$DATASETS_DIR/ifir"
mkdir -p "$DATASETS_DIR/antique"
mkdir -p "$DATASETS_DIR/bordirlines"

echo ""
echo "Dataset directories created!"
echo ""
echo "Next steps:"
echo "1. Download datasets (see DATASET_RECOMMENDATIONS.md for instructions)"
echo "2. Place TREC format files in each directory:"
echo "   - run1.txt, run2.txt (TREC format runs)"
echo "   - qrels.txt (TREC format qrels)"
echo "3. Run evaluation:"
echo "   cargo run --bin evaluate-real-world -- --datasets-dir $DATASETS_DIR"
echo ""
echo "For HuggingFace datasets, use the conversion script:"
echo "   python evals/scripts/convert_hf_to_trec.py --dataset mteb/miracl --language en --output-dir $DATASETS_DIR/miracl-en"

