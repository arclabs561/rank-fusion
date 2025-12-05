# Release v0.1.20: Fix WASM Exports

## Changes

- **Fixed**: WASM package exports - re-export wasm module functions from lib.rs
- **Updated**: E2E tests for nodejs target behavior (no async init needed)
- **Fixed**: Cross-language test Result type handling

## Technical Details

The WASM package was not exporting functions correctly because the `wasm` module wasn't re-exported from the main `lib.rs`. This fix ensures all `#[wasm_bindgen]` functions are properly accessible in JavaScript/TypeScript.

## Testing

- E2E tests updated to match nodejs target behavior
- Cross-language consistency test handles Result types correctly
- All WASM functions now properly exported

## Publishing

This release will:
1. Build WASM package with wasm-pack
2. Optimize with wasm-opt
3. Publish to npm as @arclabs561/rank-fusion@0.1.20
4. Trigger E2E tests automatically
