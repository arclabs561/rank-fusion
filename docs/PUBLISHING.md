# Publishing Guide

This guide covers publishing `rank-fusion` to:
- **crates.io** (Rust crate)
- **PyPI** (Python package)
- **npm** (WebAssembly package, if applicable)

## Prerequisites

All publishing uses **OIDC (OpenID Connect) authentication** - no manual tokens needed!

1. **crates.io account**: https://crates.io
   - Configure OIDC trusted publisher in GitHub Actions (already configured)
   - Uses `rust-lang/crates-io-auth-action@v1` for authentication

2. **PyPI account**: https://pypi.org
   - Configure trusted publisher: https://pypi.org/manage/account/publishing/
   - Uses `pypa/gh-action-pypi-publish@release/v1` (pending trusted publisher)

3. **npm account** (for WASM): https://www.npmjs.com
   - Configure OIDC trusted publisher in npm settings
   - Uses GitHub Actions OIDC (no token needed)

## Publishing Workflow

### Automated (Recommended)

1. **Create a GitHub release**:
   ```bash
   git tag v0.1.19
   git push origin v0.1.19
   ```
   Then create a release on GitHub with the same tag.

2. **GitHub Actions will automatically**:
   - Publish to crates.io
   - Publish to PyPI
   - (WASM to npm if configured)

### Manual Publishing

#### 1. Publish Rust Crate

```bash
cd rank-fusion/rank-fusion
# For automated publishing, use GitHub Actions (recommended)
# For manual publishing, you'll need a crates.io API token:
cargo publish --token YOUR_CRATES_IO_TOKEN
```

#### 2. Publish Python Package

```bash
cd rank-fusion/rank-fusion-python
uv venv
source .venv/bin/activate
uv tool install maturin

# Test first
maturin build --uv

# Publish (uses trusted publisher if configured, otherwise use token)
maturin publish --uv --username __token__ --password YOUR_PYPI_TOKEN
```

#### 3. Publish WebAssembly (npm) - Optional

If you want to publish WASM bindings:

```bash
# Install wasm-pack
curl https://rustwasm.github.io/wasm-pack/installer/init.sh -sSf | sh

# Build for npm (with wasm feature enabled)
cd rank-fusion/rank-fusion
wasm-pack build --target nodejs --out-dir pkg --scope arclabs561 -- --features wasm

# Publish to npm (uses OIDC if configured, otherwise use token)
cd pkg
npm publish --access public
```

## Version Management

- Update version in:
  - `Cargo.toml` (workspace root)
  - `rank-fusion/Cargo.toml`
  - `rank-fusion-python/pyproject.toml`
  - `rank-fusion-python/Cargo.toml`

- Use semantic versioning:
  - `MAJOR.MINOR.PATCH`
  - Breaking changes: increment MAJOR
  - New features: increment MINOR
  - Bug fixes: increment PATCH

## Pre-Publish Checklist

- [ ] All tests pass: `cargo test --workspace`
- [ ] Clippy clean: `cargo clippy --workspace -- -D warnings`
- [ ] Formatted: `cargo fmt --check --all`
- [ ] Documentation builds: `cargo doc --workspace --no-deps`
- [ ] Version numbers updated in all files
- [ ] CHANGELOG.md updated
- [ ] README.md is up to date
- [ ] Python bindings tested: `maturin develop --uv` and import works

## Post-Publish

After publishing, verify:

1. **crates.io**: https://crates.io/crates/rank-fusion
2. **PyPI**: https://pypi.org/project/rank-fusion/
3. **npm** (if applicable): https://www.npmjs.com/package/@arclabs561/rank-fusion

Test installation:

```bash
# Rust
cargo add rank-fusion

# Python
pip install rank-fusion

# npm (if applicable)
npm install @arclabs561/rank-fusion
```

