#!/bin/bash
# Setup script for creating a dataset directory structure

set -e

DATASET_NAME="${1}"
if [ -z "$DATASET_NAME" ]; then
  echo "Usage: $0 <dataset-name>"
  echo "Example: $0 msmarco"
  exit 1
fi

DATASET_DIR="./datasets/$DATASET_NAME"
mkdir -p "$DATASET_DIR"

echo "Created dataset directory: $DATASET_DIR"
echo ""
echo "Next steps:"
echo "1. Place your TREC format run files in: $DATASET_DIR/"
echo "2. Place your TREC format qrels file as: $DATASET_DIR/qrels.txt"
echo "3. Run files should have .run or .txt extension"
echo ""
echo "TREC Run Format (each line):"
echo "  query_id Q0 doc_id rank score run_tag"
echo ""
echo "TREC Qrels Format (each line):"
echo "  query_id 0 doc_id relevance"
echo ""
echo "Example run file content:"
echo "  1 Q0 doc123 1 0.95 bm25_run"
echo "  1 Q0 doc456 2 0.87 bm25_run"
echo ""
echo "Example qrels file content:"
echo "  1 0 doc123 2"
echo "  1 0 doc456 1"

