#!/bin/bash
# Monitor E2E workflow runs

echo "🔍 Monitoring E2E Workflow Runs"
echo ""

# Check rank-fusion
echo "📁 rank-fusion:"
cd /Users/arc/Documents/dev/rank-fusion
gh run list --workflow="E2E Test Published Artifacts" --limit 1 2>&1 | head -3

echo ""
echo "📁 rank-refine:"
cd /Users/arc/Documents/dev/rank-refine
gh run list --workflow="E2E Test Published Artifacts" --limit 1 2>&1 | head -3

echo ""
echo "✅ Monitoring complete"
echo ""
echo "To watch a specific run:"
echo "  cd rank-fusion && gh run watch"
