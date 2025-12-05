# Documentation Static Checks

This document describes the static checks we run to validate documentation quality.

## Automated Checks

### 1. Rust Documentation Tests (`cargo test --doc`)

**What it checks:**
- All code examples in `///` doc comments compile and run
- Examples in README.md and other docs (if extracted) are valid

**How to run locally:**
```bash
cargo test --doc --workspace
```

**CI:** Runs in `.github/workflows/ci.yml` under `docs` job.

### 2. Markdown Link Validation

**What it checks:**
- All markdown links are valid (no 404s)
- Internal file links exist
- External links are reachable

**How to run locally:**
```bash
# Install markdown-link-check
npm install -g markdown-link-check

# Check all markdown files
find . -name "*.md" -not -path "./target/*" -not -path "./.venv/*" | \
  xargs -I {} markdown-link-check {} --config .github/markdown-link-check.json
```

**CI:** Runs in `.github/workflows/docs-check.yml` (optional, continues on error).

### 3. Python Type Stub Validation

**What it checks:**
- Type stubs (`.pyi` files) are syntactically valid
- Type annotations match actual function signatures

**How to run locally:**
```bash
cd rank-fusion-python
pip install mypy
mypy rank_fusion.pyi --ignore-missing-imports
```

**CI:** Runs in `.github/workflows/docs-check.yml` (optional, continues on error).

## Manual Checks

### Code Example Validation

Before committing, verify:
- [ ] All Rust examples in README.md compile (`cargo test --doc`)
- [ ] All Python examples run without errors
- [ ] Function signatures in docs match actual implementation
- [ ] Configuration examples use correct API

### Documentation Consistency

- [ ] Terminology is consistent (e.g., "document ID" vs "doc_id")
- [ ] Links point to correct locations
- [ ] Examples use current API (not deprecated functions)

## Inspiration from Other Projects

### Rust Projects
- **Tokio**: Comprehensive doctest coverage, separate doc CI job
- **Serde**: Extensive inline examples, validates all doc comments
- **PyO3**: Python bindings with type stub validation

### Best Practices
1. **Test all doc examples**: `cargo test --doc` catches broken examples
2. **Link validation**: Prevents broken links in production docs
3. **Type stub checking**: Ensures Python bindings match Rust API
4. **Separate doc CI job**: Isolates doc issues from code issues

## Adding New Checks

To add a new static check:

1. Add the check command to `.github/workflows/docs-check.yml`
2. Document it in this file
3. Add local run instructions
4. Test in CI before enforcing (use `continue-on-error: true` initially)

