#!/bin/bash
# Download MTEB datasets (Massive Text Embedding Benchmark)
# This script provides instructions for downloading MTEB

set -e

DATASET_DIR="${1:-./datasets/mteb}"
mkdir -p "$DATASET_DIR"

echo "MTEB Dataset Download Instructions"
echo "==================================="
echo ""
echo "MTEB is available via HuggingFace and Python framework."
echo ""
echo "Python method (recommended):"
echo "  pip install mteb"
echo "  python -c \"from mteb import MTEB; benchmark = MTEB(); benchmark.download_data()\""
echo ""
echo "Or use HuggingFace:"
echo "  huggingface-cli download mteb --local-dir $DATASET_DIR"
echo ""
echo "MTEB includes:"
echo "  - 58 datasets"
echo "  - 112 languages"
echo "  - 8 task categories: retrieval, reranking, classification, clustering, etc."
echo ""
echo "For rank fusion evaluation, focus on retrieval and reranking tasks."
echo ""
echo "MTEB website: https://huggingface.co/mteb"
echo "GitHub: https://github.com/embeddings-benchmark/mteb"

