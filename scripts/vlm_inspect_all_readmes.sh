#!/bin/bash
# Visually inspect all README files using VLM (opportunistic)
# Uses API keys from environment if available

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

OUTPUT_DIR="${1:-readme_vlm_inspections}"

echo "🔍 VLM Inspection of All READMEs"
echo "================================"
echo ""

# Check for API keys
if [ -n "${ANTHROPIC_API_KEY:-}" ]; then
    echo "✅ ANTHROPIC_API_KEY found"
else
    echo "⚠️  ANTHROPIC_API_KEY not set"
fi

if [ -n "${OPENAI_API_KEY:-}" ]; then
    echo "✅ OPENAI_API_KEY found"
else
    echo "⚠️  OPENAI_API_KEY not set"
fi

if [ -z "${ANTHROPIC_API_KEY:-}" ] && [ -z "${OPENAI_API_KEY:-}" ]; then
    echo ""
    echo "⚠️  No API keys found. Will only generate screenshots."
    echo "   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable VLM inspection."
    echo ""
fi

echo ""

# Find all README.md files
readmes=(
    "$REPO_ROOT/rank-fusion/rank-fusion/README.md"
    "$REPO_ROOT/rank-refine/rank-refine/README.md"
    "$REPO_ROOT/rank-relax/README.md"
    "$REPO_ROOT/rank-eval/README.md"
)

mkdir -p "$OUTPUT_DIR"

for readme in "${readmes[@]}"; do
    if [ ! -f "$readme" ]; then
        echo "⚠️  Skipping (not found): $readme"
        continue
    fi
    
    repo_name=$(basename "$(dirname "$(dirname "$readme")")")
    echo "📄 Inspecting: $repo_name"
    echo "   File: $readme"
    
    python3 "$SCRIPT_DIR/vlm_inspect_readme.py" "$readme" "$OUTPUT_DIR/$repo_name"
    
    echo ""
done

echo "✅ All READMEs inspected!"
echo "📁 Results in: $OUTPUT_DIR"

