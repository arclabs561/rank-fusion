# Error Handling Design

This document explains the error handling philosophy and patterns used in `rank-fusion`.

## Philosophy

`rank-fusion` uses a **graceful degradation** approach rather than strict validation:

- **Most functions return `Vec<T>`** (not `Result<T, E>`) to keep the API simple
- **Invalid inputs produce empty or partial results** rather than panicking
- **Validation is opt-in** via the `validate()` function after fusion
- **Only functions that can fail meaningfully return `Result`** (e.g., `weighted_multi` with zero weights)

## Why This Design?

1. **Simplicity**: Most fusion operations are infallible (empty lists → empty results, k=0 → empty results)
2. **Performance**: No overhead from Result unwrapping in hot paths
3. **Flexibility**: Callers can choose validation level (none, warnings, strict)
4. **Consistency**: All fusion functions have the same signature pattern

## Error Handling Patterns

### Pattern 1: Graceful Degradation (Most Functions)

```rust
pub fn rrf<I: Clone + Eq + Hash>(
    results_a: &[(I, f32)],
    results_b: &[(I, f32)],
) -> Vec<(I, f32)> {
    // k=0 → empty Vec (graceful, not an error)
    if config.k == 0 {
        return Vec::new();
    }
    // ... fusion logic
}
```

**When to use**: Functions where invalid inputs have clear fallback behavior.

**Examples**: `rrf`, `combsum`, `combmnz`, `borda`, `dbsf`, `standardized`

### Pattern 2: Result for Meaningful Errors

```rust
pub fn weighted_multi<I, L>(
    lists: &[(L, f32)],
    normalize: bool,
    top_k: Option<usize>,
) -> Result<Vec<(I, f32)>, FusionError> {
    let total_weight: f32 = lists.iter().map(|(_, w)| w).sum();
    if total_weight.abs() < WEIGHT_EPSILON {
        return Err(FusionError::ZeroWeights); // Meaningful error
    }
    // ... fusion logic
}
```

**When to use**: Functions where invalid inputs indicate a programming error that should be caught.

**Examples**: `weighted_multi`, `rrf_weighted`

### Pattern 3: Validation After Fusion

```rust
let fused = rrf(&list1, &list2);
let validation = validate(&fused, false, Some(10));

if !validation.is_valid {
    eprintln!("Errors: {:?}", validation.errors);
}
```

**When to use**: Production code, debugging, quality assurance.

## Input Validation Guide

### What Gets Validated?

The `validate()` function checks:
- ✅ Sorted by score (descending)
- ✅ No duplicate document IDs
- ✅ All scores are finite (not NaN/Infinity)
- ⚠️ Non-negative scores (warning only)
- ⚠️ Result count within expected bounds

### What Doesn't Get Validated Automatically?

- ❌ Empty input lists (handled gracefully)
- ❌ k=0 in RRF (returns empty Vec)
- ❌ Non-finite input scores (RRF ignores them, score-based methods may propagate)
- ❌ Incompatible score scales (use RRF or normalize first)

### Recommended Validation Strategy

```rust
use rank_fusion::{rrf, validate, validate_finite_scores};

// 1. Perform fusion
let fused = rrf(&list1, &list2);

// 2. Validate results
let validation = validate(&fused, true, Some(20));

// 3. Handle errors
if !validation.is_valid {
    // Log errors, fallback to single retriever, etc.
    eprintln!("Fusion validation failed: {:?}", validation.errors);
    return list1; // Fallback
}

// 4. Check warnings
if !validation.warnings.is_empty() {
    // Log warnings, but continue
    eprintln!("Warnings: {:?}", validation.warnings);
}

// 5. Use fused results
return fused;
```

## Edge Case Behavior

| Input | Behavior | Validation |
|-------|----------|------------|
| Empty lists | Returns items from non-empty list(s) | ✅ Valid |
| k=0 in RRF | Returns empty Vec | ⚠️ Warning (use validate) |
| Non-finite scores | RRF ignores, score-based may propagate | ⚠️ Warning (use validate_finite_scores) |
| Duplicate IDs in input | All occurrences contribute | ✅ Valid (duplicates in output are error) |
| Zero weights | Returns `Err(FusionError::ZeroWeights)` | ❌ Error (caught by Result) |
| Mismatched list/weight lengths | Returns empty Vec (FusionStrategy) | ⚠️ Warning (use validate) |

## Migration Guide

### If You Want Strict Validation

```rust
// Before (current design)
let fused = rrf(&list1, &list2);
// May silently return empty Vec if k=0

// After (with validation)
let fused = rrf(&list1, &list2);
let validation = validate(&fused, false, None);
if !validation.is_valid {
    return Err(format!("Fusion failed: {:?}", validation.errors));
}
```

### If You Want Result-Based API

```rust
// Use functions that return Result
match weighted_multi(&lists, true, None) {
    Ok(fused) => fused,
    Err(FusionError::ZeroWeights) => {
        // Handle error
        default_weights_fusion(&lists)
    }
    Err(e) => return Err(e.into()),
}
```

## Future Considerations

Potential improvements (not implemented):
- `try_*` variants that return `Result` for all functions
- Configurable validation levels (none, warnings, strict)
- Validation at API boundaries (Python/WASM bindings already do this)

## Summary

- **Design**: Graceful degradation, validation opt-in
- **Most functions**: Return `Vec<T>`, handle edge cases gracefully
- **Some functions**: Return `Result<T, E>` for meaningful errors
- **Validation**: Use `validate()` after fusion for quality assurance
- **Production**: Always validate fusion results before using them

