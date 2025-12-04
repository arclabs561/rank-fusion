#!/bin/bash
# Download MS MARCO passage ranking dataset
# This script downloads the necessary files for MS MARCO evaluation

set -e

DATASET_DIR="${1:-./datasets/msmarco}"
mkdir -p "$DATASET_DIR"

echo "Downloading MS MARCO passage ranking dataset to $DATASET_DIR"

# MS MARCO passage ranking qrels (test set)
echo "Downloading qrels..."
curl -L "https://msmarco.blob.core.windows.net/msmarcoranking/qrels.dev.tsv" \
  -o "$DATASET_DIR/qrels.tsv" || {
  echo "Note: Direct download may require authentication. Please download manually from:"
  echo "https://microsoft.github.io/msmarco/"
}

# Convert TSV to TREC format if needed
if [ -f "$DATASET_DIR/qrels.tsv" ]; then
  echo "Converting qrels to TREC format..."
  awk '{print $1 " 0 " $2 " " $3}' "$DATASET_DIR/qrels.tsv" > "$DATASET_DIR/qrels.txt"
  echo "Qrels converted to: $DATASET_DIR/qrels.txt"
fi

echo ""
echo "Next steps:"
echo "1. Download run files from MS MARCO website or generate your own"
echo "2. Place run files in TREC format in: $DATASET_DIR/"
echo "3. Run evaluation: cargo run --bin evaluate-real-world -- --datasets-dir ./datasets"
echo ""
echo "MS MARCO website: https://microsoft.github.io/msmarco/"

