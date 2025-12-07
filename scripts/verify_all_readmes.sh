#!/bin/bash
# Verify all READMEs across rank-* repositories
# Usage: ./verify_all_readmes.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTPUT_DIR="$REPO_ROOT/readme_screenshots"

# Find all README.md files in rank-* repos
REPOS=(
    "$REPO_ROOT/README.md"
    "$REPO_ROOT/rank-fusion/README.md"
    "$(dirname "$REPO_ROOT")/rank-refine/README.md"
    "$(dirname "$REPO_ROOT")/rank-relax/README.md"
    "$(dirname "$REPO_ROOT")/rank-eval/README.md"
)

mkdir -p "$OUTPUT_DIR"

echo "Verifying READMEs across rank-* repositories..."
echo ""

for readme in "${REPOS[@]}"; do
    if [ ! -f "$readme" ]; then
        echo "⚠️  Skipping (not found): $readme"
        continue
    fi
    
    echo "📄 Processing: $readme"
    "$SCRIPT_DIR/verify_readme.sh" "$readme" "$OUTPUT_DIR"
    echo ""
done

echo "✅ All README screenshots generated in: $OUTPUT_DIR"
echo ""
echo "To verify with VLM, run:"
echo "  for img in $OUTPUT_DIR/*.png; do"
echo "    python3 $SCRIPT_DIR/verify_readme_viz.py \"\$img\" \"README verification\""
echo "  done"

