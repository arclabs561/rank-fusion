# Dataset Registry Integration

**Date:** After multi-run fusion testing and code cleanup

## Summary

Integrated the dataset registry into the evaluation pipeline to automatically include dataset metadata in evaluation results and HTML reports.

## Changes Made

### 1. **Added Dataset Metadata to Evaluation Results**
- Added `dataset_metadata: Option<DatasetEntry>` field to `DatasetEvaluationResult`
- Metadata is automatically looked up from the registry during evaluation
- Name normalization handles variations (e.g., "msmarco" vs "msmarco-passage", directory names)

### 2. **Dataset Name Normalization**
- Created `normalize_dataset_name()` function to match dataset names to registry entries
- Handles common variations:
  - Case insensitivity
  - Underscore/hyphen variations
  - Suffix removal (e.g., "-passage", "-document")
  - "trec-" prefix removal
- Falls back gracefully if dataset not found in registry

### 3. **HTML Report Enhancement**
- Added metadata section to HTML reports for each dataset
- Displays:
  - Description
  - Category and Priority
  - Domain
  - Notes (if available)
  - URL (if available, as clickable link)
- Styled with distinct background and border for visibility

### 4. **Code Changes**
- **File:** `evals/src/evaluate_real_world.rs`
  - Added `normalize_dataset_name()` helper function
  - Updated `evaluate_dataset()` to look up metadata from registry
  - Updated `generate_html_report()` to display metadata
  - Added CSS styling for metadata section

## Benefits

1. **Better Context**: Reports now include dataset descriptions, categories, and notes
2. **Easier Discovery**: Users can see dataset URLs and access information
3. **Automatic**: No manual configuration needed - metadata is looked up automatically
4. **Graceful Degradation**: Works even if dataset not found in registry

## Example Output

When evaluating a dataset like "msmarco-passage", the HTML report now includes:

```html
<div class="metadata">
    <p><strong>Description:</strong> MS MARCO Passage Ranking - Industry standard, large-scale</p>
    <p><strong>Category:</strong> General | <strong>Priority:</strong> 1 | <strong>Domain:</strong> General</p>
    <p><strong>Notes:</strong> Use MS MARCO v2 (cleaner, less biased)</p>
    <p><strong>URL:</strong> <a href="https://microsoft.github.io/msmarco/" target="_blank">https://microsoft.github.io/msmarco/</a></p>
</div>
```

## Integration Status

- ✅ Dataset metadata lookup during evaluation
- ✅ Metadata stored in evaluation results
- ✅ Metadata displayed in HTML reports
- ⚠️ Not yet used for dataset discovery/loading
- ⚠️ Not yet used for format validation
- ⚠️ Not yet used for conversion workflow suggestions

## Next Steps (Optional)

1. Use registry to guide dataset discovery (auto-suggest datasets to evaluate)
2. Use registry format information to validate dataset formats
3. Use registry to suggest conversion workflows for non-TREC datasets
4. Add metadata to JSON output for programmatic access

