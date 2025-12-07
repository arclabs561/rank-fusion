# README Verification System

System for verifying README quality using Playwright + mdpreview + VLM.

## Quick Start

```bash
# Verify a single README
./scripts/verify_readme.sh README.md

# Verify with VLM
python3 scripts/verify_readme_viz.py readme_screenshots/README.png "README for rank-fusion library"

# Verify all rank-* READMEs
./scripts/verify_all_readmes.sh
```

## What It Does

1. **Screenshot Generation**: Uses `mdpreview` to render README as HTML, then `playwright` to capture full-page screenshot
2. **VLM Verification**: Uses Claude Vision API to analyze screenshot for:
   - Visual clarity and layout
   - Structure and organization
   - Code example visibility
   - Mathematical content rendering
   - Completeness
   - Professional appearance
   - Pedagogical value

## Requirements

- `mdpreview`: `go install github.com/henrywallace/mdpreview@latest`
- `playwright`: `npm install -g playwright && npx playwright install chromium`
- `anthropic` Python package: `pip install anthropic`
- Node.js (for Playwright)

## Output

- Screenshots saved to `readme_screenshots/` directory
- VLM provides:
  - Quality score (0-100, threshold: 70)
  - Detailed feedback
  - Strengths and weaknesses
  - Improvement suggestions

## Example Output

```
============================================================
README Quality Score: 85/100
Status: ✅ PASS
============================================================

Feedback:
The README has excellent structure with clear sections, well-formatted code examples, and comprehensive documentation. Mathematical formulas render correctly. The layout is professional and easy to navigate.

Strengths:
  ✅ Clear section organization
  ✅ Well-formatted code examples
  ✅ Comprehensive API documentation
  ✅ Professional visual design

Weaknesses:
  ❌ Some sections could use more visual examples

Suggestions:
  💡 Add more diagrams for complex algorithms
  💡 Include comparison tables for different methods
```

