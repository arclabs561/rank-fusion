#!/bin/bash
# Generate dataset registry JSON file

set -e

OUTPUT="${1:-./datasets/registry.json}"

echo "Generating dataset registry..."

# Use Rust binary if available, otherwise create manually
if command -v cargo &> /dev/null; then
    cd "$(dirname "$0")/.."
    cargo run --bin list-datasets -- --json > "$OUTPUT" 2>/dev/null || {
        echo "Note: Registry generation requires compiled binary"
        echo "Run: cargo build -p rank-fusion-evals"
    }
else
    echo "Cargo not found. Please install Rust to generate registry."
    exit 1
fi

echo "Registry written to: $OUTPUT"

