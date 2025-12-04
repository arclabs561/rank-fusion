# Evaluation Analysis and Refinements

## Current Status: 22/25 scenarios correct (88%)

## Key Insights from Running Evals

### 1. Method Similarities Are Expected

Many methods are mathematically similar, which is correct behavior:
- `additive_multi_task` with Z-score normalization is similar to `standardized`
- Both use z-score, but `standardized` clips while `additive_multi_task` doesn't
- This makes some scenarios hard to differentiate

### 2. Weighted Methods Can Dominate

`additive_multi_task_1_20` (with 20× weight on list B) can dominate scenarios where:
- List B strongly favors relevant documents
- The weight ratio is extreme (1:20)
- This is correct behavior, but makes scenarios hard to design

### 3. Tie-Breaking Improvements

Improved tie-breaking logic:
- Primary: nDCG@5 (handles graded relevance)
- Tie-breaker 1: Average Precision (overall quality)
- Tie-breaker 2: Precision@1 (top result)
- Tie-breaker 3: MRR (first relevant)

This provides more nuanced winner selection.

## Remaining Failing Scenarios (3)

### 1. `isr_steeper_decay` - Expected: `isr`, Actual: `combmnz`
**Analysis**: ISR and CombMNZ are both performing well. The scenario may need to be more specific to ISR's steeper decay behavior.

### 2. `standardized_tight_clipping` - Expected: `standardized_tight`, Actual: `additive_multi_task_1_20`
**Analysis**: The 20× weight on list B is dominating. Need to balance lists or reduce weight advantage.

### 3. `standardized_outlier_robustness` - Expected: `standardized`, Actual: `additive_multi_task_1_20`
**Analysis**: Similar issue - 20× weight dominates. Need to balance lists or adjust expected winner.

## Refinements Applied

1. **Improved tie-breaking**: nDCG@5 → AP → P@1 → MRR
2. **Refined scenario data**: Better scores to highlight intended behavior
3. **Better documentation**: Clear comments explaining expected winners
4. **Balanced lists**: Reduced weight advantage in scenarios

## Recommendations

1. **Accept Valid Ties**: When methods are mathematically similar, accept ties
2. **Focus on Clear Differentiation**: Each scenario should have a clear reason why one method wins
3. **Consider Method Similarities**: Document when methods are expected to perform similarly
4. **Test Edge Cases**: Add more scenarios for edge cases

## Next Steps

1. Continue refining the 3 failing scenarios
2. Consider adjusting expected winners if methods are truly equivalent
3. Add more scenarios that clearly differentiate methods
4. Document when ties are acceptable

