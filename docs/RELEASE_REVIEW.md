# Test Release Review

## Release Created

- **Tag**: `v0.1.20-test`
- **Purpose**: Validate publishing workflows
- **Status**: Created locally

## Next Steps

1. **Push tag to GitHub:**
   ```bash
   git push origin v0.1.20-test
   ```

2. **Create GitHub Release:**
   - Go to: https://github.com/arclabs561/rank-fusion/releases/new
   - Select tag: `v0.1.20-test`
   - Title: `Test Release - Workflow Validation`
   - Description: Copy from `TEST_RELEASE_NOTES.md`
   - Click "Publish release"

3. **Monitor Workflows:**
   - Actions: https://github.com/arclabs561/rank-fusion/actions
   - Watch for:
     - `Publish` workflow
     - `Publish WASM` workflow

## Expected Workflow Execution

### Validation Job
- ✅ Version consistency check
- ✅ Run tests
- ✅ Run clippy
- ✅ Check formatting

### Publish Jobs (after validation)
- ✅ Publish crate to crates.io
- ✅ Publish Python to PyPI
- ✅ Publish WASM to npm

## Review Checklist

- [ ] Tag pushed to GitHub
- [ ] Release created on GitHub
- [ ] Workflows triggered
- [ ] Validation job passes
- [ ] Publish jobs execute (may fail if publishers not configured)
- [ ] Review workflow logs for any issues

## Notes

- If PyPI trusted publisher not configured, Python publish will fail (expected)
- If npm OIDC not configured, WASM publish will fail (expected)
- Validation should always pass if code is correct
