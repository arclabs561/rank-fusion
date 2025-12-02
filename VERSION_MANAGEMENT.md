# Version Management Guide

This guide explains how to update versions across all files in the `rank-fusion` workspace.

## Files That Need Version Updates

When releasing a new version, update the version in these files:

1. **`Cargo.toml`** (workspace root) - Main version definition
2. **`rank-fusion/Cargo.toml`** - Uses workspace version (no change needed if using `version.workspace = true`)
3. **`rank-fusion-python/pyproject.toml`** - Python package version
4. **`rank-fusion-python/Cargo.toml`** - Uses workspace version (no change needed if using `version.workspace = true`)

## Version Update Process

### Step 1: Update Workspace Version

Edit `Cargo.toml` at the root:

```toml
[workspace.package]
version = "0.1.20"  # Update this
```

### Step 2: Update Python Package Version

Edit `rank-fusion-python/pyproject.toml`:

```toml
[project]
version = "0.1.20"  # Update this to match workspace version
```

### Step 3: Verify Version Consistency

Run the version check script (or let CI do it):

```bash
# Manual check
ROOT_VERSION=$(grep '^version = ' Cargo.toml | cut -d'"' -f2)
CRATE_VERSION=$(grep '^version = ' rank-fusion/Cargo.toml | grep -o '".*"' | tr -d '"' || echo "workspace")
PYTHON_VERSION=$(grep '^version = ' rank-fusion-python/pyproject.toml | cut -d'"' -f2)

echo "Root: $ROOT_VERSION"
echo "Crate: $CRATE_VERSION"
echo "Python: $PYTHON_VERSION"
```

Or let CI validate it automatically when you create a release.

### Step 4: Commit and Tag

```bash
git add Cargo.toml rank-fusion-python/pyproject.toml
git commit -m "Bump version to 0.1.20"
git tag v0.1.20
git push origin master
git push origin v0.1.20
```

## Semantic Versioning

Follow [Semantic Versioning](https://semver.org/):

- **MAJOR** (1.0.0): Breaking API changes
- **MINOR** (0.1.0): New features, backward compatible
- **PATCH** (0.0.1): Bug fixes, backward compatible

### Examples

- `0.1.19` → `0.1.20`: Patch release (bug fix)
- `0.1.19` → `0.2.0`: Minor release (new feature)
- `0.1.19` → `1.0.0`: Major release (breaking change)

## Automated Version Checking

The CI workflow automatically checks version consistency before publishing:

- **Pre-publish validation**: Runs tests, clippy, formatting, and version checks
- **Version mismatch detection**: Fails if versions don't match across files
- **Prevents broken releases**: Won't publish if validation fails

## Troubleshooting

### "Version mismatch" error

If CI fails with a version mismatch:

1. Check all files listed above have the same version
2. Ensure workspace inheritance is working (`version.workspace = true` in child crates)
3. Verify no typos in version strings

### Workspace version not inherited

If a crate isn't using the workspace version:

1. Check `Cargo.toml` has `version.workspace = true` (not `version = "..."`)
2. Verify the crate is a workspace member
3. Ensure `[workspace.package]` is defined in root `Cargo.toml`

## Best Practices

1. **Always update both files**: Root `Cargo.toml` and `pyproject.toml`
2. **Use workspace inheritance**: Let child crates inherit from workspace
3. **Test before releasing**: Run `cargo test --workspace` and `cargo clippy`
4. **Check CI passes**: Ensure version check passes before tagging
5. **Document breaking changes**: Update CHANGELOG.md for major/minor releases

