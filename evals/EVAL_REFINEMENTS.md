# Evaluation Refinements Based on Running Evals

## Summary

After running the evals multiple times and reviewing results, we've identified several areas for refinement:

### Current Status: 22/25 scenarios correct (88%)

### Key Findings

1. **Additive Multi-Task with Z-Score is Very Robust**
   - `additive_multi_task_1_20` and `additive_multi_task_1_1` are winning many scenarios
   - Z-score normalization (default) is very robust to outliers
   - This is actually correct behavior, but scenarios need refinement

2. **Tie-Breaking Logic Improved**
   - Added better tie-breaking: nDCG@5 → AP → P@1 → MRR
   - More nuanced winner selection

3. **Scenarios Need Better Differentiation**
   - Some scenarios are too similar, causing multiple methods to tie
   - Need more specific scenarios that clearly differentiate methods

## Failing Scenarios (3)

### 1. `isr_steeper_decay` - Expected: `isr`, Actual: `combmnz`
**Issue**: ISR's steeper decay isn't differentiating enough from CombMNZ's overlap multiplier.

**Refinement Strategy**:
- Make the scenario more specific to ISR's behavior
- Ensure rel1 appears at rank 0 in one list but not the other
- ISR should emphasize top ranks more than CombMNZ

### 2. `standardized_tight_clipping` - Expected: `standardized_tight`, Actual: `additive_multi_task_1_20`
**Issue**: Additive multi-task with Z-score is winning because it also uses z-score normalization.

**Refinement Strategy**:
- Make the scenario more specific to tight clipping benefits
- Ensure the outlier's z-score falls between -2 and -3 (or 2 and 3)
- Tight clipping should specifically help in this range

### 3. `standardized_outlier_robustness` - Expected: `standardized`, Actual: `additive_multi_task_1_20`
**Issue**: Both methods use z-score normalization, so they perform similarly.

**Refinement Strategy**:
- Focus on the difference: standardized uses clipping, additive_multi_task doesn't clip after normalization
- Create scenario where clipping specifically helps
- Or adjust expected winner to reflect that both methods are robust

## Refinements Made

### 1. Improved Tie-Breaking Logic
```rust
// Before: Only nDCG@5
// After: nDCG@5 → AP → P@1 → MRR
```

### 2. Refined Scenario Data
- **rank_based_beats_outlier**: Extended lists to make Borda's penalty for inconsistent rankings more pronounced
- **standardized_tight_clipping**: Adjusted scores to better highlight tight clipping benefits
- **standardized_outlier_robustness**: Made outlier more extreme to differentiate methods

### 3. Better Documentation
- Added detailed comments explaining why each scenario should favor a specific method
- Documented the mathematical reasoning behind expected winners

## Recommendations

1. **Accept Some Ties as Valid**
   - When multiple methods achieve the same nDCG@5, they may all be valid winners
   - Consider scenarios where "any of these methods" is acceptable

2. **Focus on Clear Differentiation**
   - Each scenario should have a clear mathematical reason why one method wins
   - Avoid scenarios where multiple methods are equally valid

3. **Consider Method Similarities**
   - `additive_multi_task` with Z-score is similar to `standardized` in robustness
   - This is expected behavior, not a bug

4. **Add More Edge Cases**
   - Empty lists
   - Single-element lists
   - All scores equal
   - Complete disagreement

## Next Steps

1. Continue refining the 3 failing scenarios
2. Add more scenarios that clearly differentiate methods
3. Consider adding scenarios that test edge cases
4. Document expected behavior when methods tie

## Evaluation Metrics

Current evaluation uses:
- **Primary**: nDCG@5 (handles graded relevance, emphasizes top results)
- **Tie-breaker 1**: Average Precision (overall ranking quality)
- **Tie-breaker 2**: Precision@1 (top result quality)
- **Tie-breaker 3**: MRR (first relevant result)

This provides a comprehensive view of ranking quality while emphasizing top results.

