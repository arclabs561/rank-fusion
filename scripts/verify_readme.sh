#!/bin/bash
# Verify README quality using Playwright + mdpreview + VLM
# Usage: ./verify_readme.sh <readme_path> [output_dir]

set -euo pipefail

README_FILE="${1:-}"
OUTPUT_DIR="${2:-readme_screenshots}"

if [ -z "$README_FILE" ]; then
    echo "Usage: $0 <readme_path> [output_dir]"
    exit 1
fi

if [ ! -f "$README_FILE" ]; then
    echo "Error: README file not found: $README_FILE"
    exit 1
fi

# Check dependencies
if ! command -v mdpreview &> /dev/null; then
    echo "Error: mdpreview not found. Install with: go install github.com/henrywallace/mdpreview@latest"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

# Start mdpreview server in background
PORT=8080
mdpreview -addr ":$PORT" "$README_FILE" &
MDPREVIEW_PID=$!

# Wait for server to start
sleep 3

# Generate screenshot with Playwright
SCREENSHOT_FILE="$OUTPUT_DIR/$(basename "$README_FILE" .md).png"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

node "$SCRIPT_DIR/screenshot_readme.js" "http://localhost:$PORT" "$SCREENSHOT_FILE"

# Kill mdpreview
kill $MDPREVIEW_PID 2>/dev/null || true

echo "✅ Screenshot generated: $SCREENSHOT_FILE"
echo "Run VLM verification with:"
echo "  python3 scripts/verify_readme_viz.py '$SCREENSHOT_FILE' 'README for rank-fusion library'"

