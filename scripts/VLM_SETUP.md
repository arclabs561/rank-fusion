# VLM Visual Inspection Setup

## Quick Start

The VLM inspection system opportunistically uses API keys from environment variables to provide visual critiques of rendered README pages.

### 1. Install Dependencies

```bash
# Python dependencies (will be auto-installed via PEP 723)
# The scripts use inline dependencies, so just run:
uv run vlm_inspect_readme.py <readme_path>
```

### 2. Set API Keys (Optional)

```bash
# Claude (preferred - better for technical documentation)
export ANTHROPIC_API_KEY="sk-ant-..."

# Or OpenAI (fallback)
export OPENAI_API_KEY="sk-..."
```

**Note**: API keys are optional. The system will still generate screenshots without keys, just without VLM critique.

### 3. Run Inspection

```bash
# Single README
python3 scripts/vlm_inspect_readme.py rank-fusion/rank-fusion/README.md

# All READMEs
./scripts/vlm_inspect_all_readmes.sh
```

## How It Works

1. **Renders markdown** using `mdpreview` (GitHub-flavored)
2. **Takes screenshot** using Playwright (full-page)
3. **Sends to VLM** (if API keys available):
   - Claude Vision (preferred)
   - GPT-4 Vision (fallback)
4. **Provides critique** on visual quality, rendering, layout

## Opportunistic Behavior

The system gracefully degrades:
- ✅ **With API keys**: Full VLM critique + screenshot
- ✅ **Without API keys**: Screenshot only (no critique)

This makes it safe for CI/CD - it won't fail if keys aren't set.

## Cost

- **Screenshots**: Free (local)
- **VLM API**: ~$0.01-0.05 per README
- **4 READMEs**: ~$0.04-0.20 per run

## Example Output

```
📸 Taking screenshot of README.md...
✅ Screenshot saved: readme_screenshots/README_screenshot.png
   Image size: 1920x4500

🔍 Inspecting with VLM (anthropic)...
   Trying Claude Vision...

✅ VLM Critique (Claude):
======================================================================
Visual Quality Score: 8.5/10

The README renders excellently with:
- Clear section hierarchy
- Well-formatted code blocks
- Mathematical formulas render correctly
- Visualizations are appropriately sized
- Professional appearance

Minor issues:
- Some code blocks could use more spacing
- Image on line 260 slightly too large for mobile

Recommendations:
- Add whitespace around formulas
- Consider responsive image sizing
======================================================================

💾 Critique saved: readme_screenshots/README_vlm_critique.txt
```

## Integration

### CI/CD Example

```yaml
- name: VLM Inspect READMEs
  env:
    ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
  run: |
    ./scripts/vlm_inspect_all_readmes.sh
```

The script won't fail if the secret isn't set - it just skips VLM critique.

## Troubleshooting

### No API keys available
```
⚠️  No API keys found in environment variables.
   Set ANTHROPIC_API_KEY or OPENAI_API_KEY to enable VLM inspection.
   Falling back to screenshot generation only.
```
**Solution**: Set environment variables or continue without (screenshots still work)

### mdpreview not found
```
❌ Error: mdpreview not found
```
**Solution**: `go install github.com/henrywallace/mdpreview@latest`

### Playwright not installed
```
Error: Cannot find module 'playwright'
```
**Solution**: `cd scripts && npm install playwright && npx playwright install chromium`

## Files

- `vlm_inspect_readme.py` - Main inspection script (opportunistic API keys)
- `vlm_inspect_all_readmes.sh` - Batch inspection script
- `verify_readme_viz.py` - Updated to be opportunistic (backward compatible)
- `screenshot_readme.js` - Playwright screenshot script

