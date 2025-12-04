# Test Release Guide

## Summary

After reviewing GitHub best practices and similar projects, our workflows are production-ready and follow industry standards.

## Workflow Review Results

✅ **All workflows follow best practices:**
- OIDC authentication (no tokens needed)
- Proper permissions (`id-token: write`, `contents: read`)
- Version consistency checks
- Validation before publishing
- Consistent patterns across all repos

## Creating a Test Release

To test the publishing workflows:

### Option 1: Using GitHub UI (Recommended)

1. **Commit and push current changes:**
   ```bash
   git add .
   git commit -m "chore: Add publishing workflow improvements"
   git push origin master
   ```

2. **Create a test release tag:**
   ```bash
   git tag v0.1.20-test
   git push origin v0.1.20-test
   ```

3. **Create GitHub Release:**
   - Go to https://github.com/arclabs561/rank-fusion/releases/new
   - Select tag: `v0.1.20-test`
   - Title: `Test Release - Workflow Validation`
   - Description: Use content from `TEST_RELEASE_NOTES.md`
   - Click "Publish release"

4. **Monitor Workflows:**
   - Go to https://github.com/arclabs561/rank-fusion/actions
   - Watch the "Publish" and "Publish WASM" workflows

### Option 2: Using GitHub CLI

```bash
# After committing and pushing
gh release create v0.1.20-test \
  --title "Test Release - Workflow Validation" \
  --notes-file TEST_RELEASE_NOTES.md
```

## Expected Workflow Behavior

The workflows will:

1. **Validate:**
   - Check version consistency
   - Run tests
   - Run clippy
   - Check formatting

2. **Publish (if validation passes):**
   - Crate to crates.io (OIDC)
   - Python to PyPI (trusted publisher)
   - WASM to npm (OIDC)

## Notes

- If trusted publishers aren't configured, PyPI publish will fail (but workflow will run)
- If npm OIDC isn't configured, npm publish will fail (but workflow will run)
- The validation step will always run and verify the process

## Next Steps After Test

1. Verify workflows run successfully
2. Check that validation passes
3. Configure trusted publishers if needed
4. Create actual release when ready
