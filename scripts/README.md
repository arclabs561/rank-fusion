# README Verification Scripts

Scripts for verifying README quality using Playwright + mdpreview + VLM.

## Setup

1. **Install mdpreview:**
   ```bash
   go install github.com/henrywallace/mdpreview@latest
   ```

2. **Install Playwright:**
   ```bash
   npm install -g playwright
   npx playwright install chromium
   ```

3. **Install Python dependencies:**
   ```bash
   pip install anthropic
   ```

4. **Set API key (optional, can use env var):**
   ```bash
   export ANTHROPIC_API_KEY=your_key_here
   ```

## Usage

### Verify Single README

```bash
# Generate screenshot
./scripts/verify_readme.sh README.md

# Verify with VLM
python3 scripts/verify_readme_viz.py readme_screenshots/README.png "README for rank-fusion library"
```

### Verify All READMEs

```bash
# Generate screenshots for all rank-* READMEs
./scripts/verify_all_readmes.sh

# Verify all with VLM (batch)
for img in readme_screenshots/*.png; do
  python3 scripts/verify_readme_viz.py "$img" "README verification"
done
```

## What It Checks

The VLM verification evaluates:
- **Visual Clarity**: Layout, readability, organization
- **Structure**: Section organization, headings
- **Code Examples**: Visibility and formatting
- **Mathematical Content**: Formula rendering (if any)
- **Completeness**: Comprehensiveness
- **Professional Appearance**: Polish and production-readiness
- **Pedagogical Value**: Helpfulness for understanding/using the library

## Output

- Screenshots saved to `readme_screenshots/`
- VLM verification provides:
  - Score (0-100)
  - Detailed feedback
  - Strengths and weaknesses
  - Improvement suggestions

