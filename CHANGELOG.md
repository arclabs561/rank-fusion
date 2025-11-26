# Changelog

## [0.1.3] - Unreleased

### Added
- Property-based tests using proptest
- Tests for commutativity, score positivity, output bounds

### Fixed
- Fixed `combsum_basic` test case

## [0.1.2] - 2024-11-26

### Fixed
- Removed overstated performance claims from README
- Corrected `rrf_into` docs (still allocates HashMap internally)
- Added edge case tests for empty inputs, duplicate IDs

### Added
- `rrf_score_formula` test verifying RRF math
- `combsum_basic` standalone test
- `weighted_skewed` test for both weight directions
- `both_empty` test
- `duplicate_ids_in_same_list` test

## [0.1.1] - 2024-11-26

### Added
- DESIGN.md with algorithm documentation

## [0.1.0] - 2024-11-26

### Added
- Initial release
- RRF (Reciprocal Rank Fusion)
- CombSUM, CombMNZ
- Borda count
- Weighted fusion with normalization

