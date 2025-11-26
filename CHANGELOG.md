# Changelog

## [0.1.7] - 2025-11-26

### Fixed
- **NaN handling**: Sorting now uses `total_cmp` for deterministic NaN placement
  (learned from rank-refine audit)

### Added
- `#[must_use]` on all pure functions (12 functions)
- 8 new property tests:
  - `nan_does_not_corrupt_sorting`
  - `infinity_handled_gracefully`
  - `output_always_sorted`
  - `unique_ids_in_output`
  - `combsum_scores_nonnegative`
  - `equal_weights_symmetric`
  - `rrf_score_bounded`
  - `empty_list_preserves_ids`
- Internal `sort_scored_desc` helper to centralize sorting logic

## [0.1.6] - 2025-11-26
- Added MSRV 1.70 to Cargo.toml
- Added CI caching with `Swatinem/rust-cache`
- Added MSRV CI job
- Added cross-link to rank-refine in README
- Removed unused `RankedList` newtype

## [0.1.5] - 2025-11-26
- Added `borda_multi`, `combsum_multi`, `combmnz_multi`, `weighted_multi` for 3+ lists
- Two-list functions now delegate to multi variants internally

## [0.1.4] - 2025-11-26
- Simplified docs
- Property tests

## [0.1.3] - 2025-11-26
- CI setup
- CHANGELOG

## [0.1.2] - 2025-11-26
- Edge case tests

## [0.1.0] - 2025-11-26
- Initial release
