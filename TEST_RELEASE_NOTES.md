# Test Release - Workflow Validation

This is a test release to validate the publishing workflows.

## Changes Tested

- ✅ Publishing workflow validation
- ✅ Version consistency checks
- ✅ OIDC authentication
- ✅ Python bindings publishing
- ✅ WASM bindings publishing

## Workflow Status

This release will trigger:
1. Validation job (tests, clippy, formatting)
2. Crate publishing to crates.io
3. Python publishing to PyPI
4. WASM publishing to npm

## Notes

This is a test release and may not actually publish to registries if:
- PyPI trusted publishers are not configured
- npm OIDC is not configured
- Version already exists in registries

The workflow will still run and validate the process.
