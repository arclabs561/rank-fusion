# VLM Visual Inspection of READMEs

Automated visual inspection of rendered README pages using Vision Language Models (VLM).

## Overview

This system:
1. Renders markdown files using `mdpreview` (GitHub-flavored markdown)
2. Takes full-page screenshots using Playwright
3. Uses VLM (Claude Vision or GPT-4 Vision) to critique visual appearance
4. Provides specific feedback on rendering quality

## Opportunistic API Key Usage

The system **opportunistically** uses API keys from environment variables:
- If `ANTHROPIC_API_KEY` is set → uses Claude Vision (preferred)
- If `OPENAI_API_KEY` is set → uses GPT-4 Vision (fallback)
- If neither is set → only generates screenshots (no VLM critique)

**No API keys required** - the system gracefully degrades to screenshot-only mode.

## Setup

### 1. Install Dependencies

```bash
# mdpreview (Go tool)
go install github.com/henrywallace/mdpreview@latest

# Playwright (Node.js)
cd scripts
npm install playwright
npx playwright install chromium
```

### 2. Set API Keys (Optional)

```bash
# Claude (preferred)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI (fallback)
export OPENAI_API_KEY="sk-..."
```

## Usage

### Inspect Single README

```bash
python3 scripts/vlm_inspect_readme.py rank-fusion/rank-fusion/README.md
```

### Inspect All READMEs

```bash
./scripts/vlm_inspect_all_readmes.sh
```

### Custom Output Directory

```bash
python3 scripts/vlm_inspect_readme.py README.md custom_output/
```

## Output

For each README, generates:
1. **Screenshot**: `{readme_name}_screenshot.png` - Full-page rendered view
2. **VLM Critique**: `{readme_name}_vlm_critique.txt` - Detailed visual feedback (if API keys available)

## VLM Critique Focus Areas

The VLM analyzes:
- ✅ Visual clarity and readability
- ✅ Mathematical formulas rendering (LaTeX/MathJax)
- ✅ Code blocks formatting and syntax highlighting
- ✅ Image/visualization placement and quality
- ✅ Overall professional appearance
- ✅ Layout issues or broken elements
- ✅ Text readability and typography

## Example Output

```
✅ VLM Critique (Claude):
======================================================================
Visual Quality Score: 8.5/10

Strengths:
- Clean, professional layout
- Mathematical formulas render correctly
- Code blocks have proper syntax highlighting
- Visualizations are well-placed and high quality

Issues:
- Formula on line 237 could use more spacing
- Image on line 260 is slightly too large for mobile view
- Code block on line 45 has inconsistent indentation

Recommendations:
- Add more whitespace around formulas
- Consider responsive image sizing
- Standardize code block formatting
======================================================================
```

## Integration with CI/CD

The system is designed to work in CI/CD pipelines:

```yaml
# .github/workflows/readme-check.yml
- name: VLM Inspect READMEs
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    ./scripts/vlm_inspect_all_readmes.sh
```

**Note**: API keys are optional. The system will still generate screenshots even without keys.

## Cost Considerations

- **Screenshots**: Free (local generation)
- **VLM API calls**: ~$0.01-0.05 per README (depends on image size)
- **Total for 4 READMEs**: ~$0.04-0.20 per run

## Troubleshooting

### mdpreview not found
```bash
go install github.com/henrywallace/mdpreview@latest
```

### Playwright not installed
```bash
cd scripts
npm install playwright
npx playwright install chromium
```

### API key errors
- Check key is set: `echo $ANTHROPIC_API_KEY`
- Verify key format (Claude: `sk-ant-...`, OpenAI: `sk-...`)
- Check network connectivity

### Screenshot timeout
- Increase timeout in `screenshot_readme.js`
- Check mdpreview server is running: `curl http://localhost:8080`

## Files

- `vlm_inspect_readme.py` - Main inspection script
- `vlm_inspect_all_readmes.sh` - Batch inspection script
- `screenshot_readme.js` - Playwright screenshot script
- `verify_readme.sh` - Original verification script (screenshot only)

## Comparison

| Feature | verify_readme.sh | vlm_inspect_readme.py |
|---------|------------------|----------------------|
| Screenshot | ✅ | ✅ |
| VLM Critique | ❌ | ✅ (optional) |
| API Keys Required | ❌ | ❌ (optional) |
| Batch Processing | ❌ | ✅ |

