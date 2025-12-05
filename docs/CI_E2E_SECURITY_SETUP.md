# CI E2E & Security Setup

## Overview

This repository now includes comprehensive CI workflows with:
- ✅ End-to-end (E2E) validation tests
- ✅ Security scanning and audits
- ✅ Best practices checks
- ✅ Quality assurance

## CI Workflows

### 1. Main CI Workflow (`.github/workflows/ci.yml`)

**Jobs:**
- **test**: Runs tests on multiple OS/rust versions
- **msrv**: Minimum Supported Rust Version check
- **clippy**: Linting with strict warnings
- **fmt**: Code formatting check
- **docs**: Documentation build and doctest
- **security**: Security and dependency checks
- **e2e-local**: E2E validation tests
- **quality**: Best practices and quality checks
- **mutation**: Mutation testing

### 2. Security Audit Workflow (`.github/workflows/security-audit.yml`)

**Runs:**
- On every push/PR
- Weekly schedule (Mondays)

**Checks:**
- `cargo-audit`: Vulnerability scanning
- `cargo-deny`: License and dependency checks
- Duplicate dependency detection
- Security report generation

### 3. E2E Published Workflow (`.github/workflows/e2e-published.yml`)

**Runs:**
- After publishing workflows complete
- Tests published packages (Rust, Python, WASM)

## E2E Tests

Located in `test-e2e-local/`, these tests simulate real-world usage:

1. **test-fusion-basic**: All rank-fusion algorithms
2. **test-fusion-eval-integration**: rank-fusion + rank-eval integration
3. **test-refine-basic**: rank-refine functionality
4. **test-eval-basic**: rank-eval functionality
5. **test-full-pipeline**: Complete RAG pipeline

### Running E2E Tests Locally

```bash
# Run all E2E tests
for bin in test-fusion-basic test-fusion-eval-integration test-refine-basic test-eval-basic test-full-pipeline; do
    cargo run -p test-e2e-local --bin $bin
done
```

## Security Tools

### cargo-audit

Scans dependencies for known vulnerabilities from the RustSec advisory database.

```bash
cargo install cargo-audit --locked
cargo audit
```

### cargo-deny

Comprehensive dependency checking:
- Security advisories
- License compliance
- Dependency bans
- Duplicate detection

Configuration: `.cargo-deny.toml`

```bash
# Install
curl -L https://github.com/rustsec/cargo-deny/releases/latest/download/cargo-deny-x86_64-unknown-linux-musl.tar.gz | tar -xz
sudo mv cargo-deny /usr/local/bin/

# Run checks
cargo-deny check advisories
cargo-deny check licenses
cargo-deny check bans
```

## Best Practices Checks

The CI workflow includes automated checks for:

1. **Unsafe Code**: Detects `unsafe` keyword usage
2. **Error Handling**: Checks for `unwrap()`/`expect()` usage
3. **Documentation**: Verifies documentation coverage
4. **TODO/FIXME**: Tracks incomplete work
5. **Cargo.toml Metadata**: Validates package metadata

## Quality Metrics

### Code Quality
- ✅ Clippy with strict warnings (`-D warnings`)
- ✅ Rustfmt formatting
- ✅ Documentation warnings as errors
- ✅ Mutation testing

### Security
- ✅ Vulnerability scanning (cargo-audit)
- ✅ License compliance (cargo-deny)
- ✅ Dependency duplicate detection
- ✅ Weekly security audits

### Testing
- ✅ Unit tests
- ✅ Integration tests
- ✅ E2E validation tests
- ✅ Doc tests
- ✅ Mutation tests

## Running Checks Locally

### Security Audit
```bash
cargo audit
cargo-deny check
```

### Quality Checks
```bash
cargo clippy --workspace -- -D warnings
cargo fmt --check --all
cargo test --workspace
```

### E2E Tests
```bash
cargo run -p test-e2e-local --bin test-fusion-basic
# ... etc
```

## CI Status

All checks must pass before merging:
- ✅ Tests pass
- ✅ Clippy clean
- ✅ Formatted
- ✅ Docs build
- ✅ Security audit passes
- ✅ E2E tests pass
- ✅ Quality checks pass

## Future Enhancements

- [ ] Dependabot integration
- [ ] Code coverage reporting
- [ ] Performance benchmarks in CI
- [ ] Fuzzing in CI
- [ ] SBOM generation

