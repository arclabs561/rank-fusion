#!/bin/bash
# Download MIRACL dataset (multilingual IR)
# This script provides instructions for downloading MIRACL

set -e

DATASET_DIR="${1:-./datasets/miracl}"
mkdir -p "$DATASET_DIR"

echo "MIRACL Dataset Download Instructions"
echo "====================================="
echo ""
echo "MIRACL is available via HuggingFace Datasets."
echo ""
echo "Python method (recommended):"
echo "  pip install datasets"
echo "  python -c \"from datasets import load_dataset; ds = load_dataset('miracl/miracl', 'en'); ds.save_to_disk('$DATASET_DIR/en')\""
echo ""
echo "Or use the HuggingFace CLI:"
echo "  huggingface-cli download miracl/miracl --local-dir $DATASET_DIR"
echo ""
echo "MIRACL includes 18 languages:"
echo "  ar, de, en, es, fa, fi, fr, hi, id, it, ja, ko, nl, pl, pt, ru, sw, th, zh"
echo ""
echo "After downloading, you'll need to convert to TREC format for evaluation."
echo ""
echo "MIRACL website: https://project-miracl.github.io/"
echo "HuggingFace: https://huggingface.co/datasets/mteb/miracl"

