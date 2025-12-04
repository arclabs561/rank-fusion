# Changelog

## [Unreleased]

### Added
- **Standardized Fusion (ERANK-style)**: Z-score normalization with configurable clipping
  - `standardized(&a, &b)` — simple API with default clipping [-3.0, 3.0]
  - `standardized_with_config(&a, &b, config)` — customizable clipping range and top_k
  - `standardized_multi` for 3+ lists
  - `FusionMethod::Standardized` variant
  - `StandardizedConfig` with `clip_range` and `top_k` options
  - More robust to outliers than min-max normalization
  - Handles negative scores naturally (z-score works with any distribution)
  - Based on ERANK paper showing 2-5% NDCG improvement when distributions differ
- **Additive Multi-Task Fusion (ResFlow-style)**: Weighted additive fusion for multi-task ranking
  - `additive_multi_task(&a, &b, config)` — configurable weights and normalization
  - `additive_multi_task_multi` for 3+ lists
  - `FusionMethod::AdditiveMultiTask` variant
  - `AdditiveMultiTaskConfig` with weights, normalization method, and top_k
  - Optimized for e-commerce ranking (CTR + CTCVR with 1:20 weight ratio)
  - Supports all normalization methods (ZScore, MinMax, Sum, Rank, None)
  - Based on ResFlow paper showing additive outperforms multiplicative for multi-task ranking
- **Fine-Grained Scoring (0-10 integer scale)** in `rank-refine`:
  - `rerank_fine_grained()` — maps similarity scores to 0-10 integer scale
  - `FineGrainedConfig` with score range, temperature, and normalization options
  - Better discrimination than binary classification
  - Useful for LLM-based reranking with integer outputs
- 13 new evaluation scenarios testing:
  - Distribution mismatch handling
  - Outlier robustness
  - Negative score handling
  - Extreme weight ratios (1:100)
  - E-commerce funnel scenarios
- 2 new example files:
  - `examples/standardized_fusion.rs` — demonstrates standardized fusion
  - `examples/additive_multi_task.rs` — demonstrates ResFlow-style multi-task fusion
- Comprehensive test coverage:
  - 8 new unit tests for standardized fusion
  - 6 new unit tests for additive multi-task fusion
  - 5 new integration tests for standardized fusion
  - 5 new integration tests for additive multi-task fusion
  - 5 new integration tests for fine-grained scoring
  - Total: 169 tests passing (113 unit + 22 integration + 34 rank-refine)

### Changed
- Evaluation system now includes new methods in all scenarios
- HTML evaluation report updated with method descriptions
- Benchmarks added for `standardized` and `additive_multi_task`

## [0.1.18] - 2025-11-27

### Added
- Property tests for CombMNZ and DBSF commutativity
- Total of 29 property tests now covering all algorithms

## [0.1.16] - 2025-11-27

### Added
- **ISR (Inverse Square Root Rank)** fusion: gentler decay than RRF
  - `isr(&a, &b)` — simple API with k=1 default
  - `isr_with_config(&a, &b, config)` — customizable k and top_k
  - `isr_multi` for 3+ lists
  - `FusionMethod::Isr` and `FusionMethod::isr()` / `isr_with_k(k)`
  - Formula: `score(d) = Σ 1/sqrt(k + rank)` where rank is 0-indexed
  - Use when lower ranks should contribute more relative to top positions
- 10 new ISR tests (unit + property tests)

## [0.1.15] - 2025-11-27

### Changed
- **API consistency**: `rrf()` now takes references and uses default config
  - New: `rrf(&a, &b)` — simple API with k=60 default
  - New: `rrf_with_config(&a, &b, config)` — customizable k and top_k
  - Matches `combsum`, `borda`, `dbsf` pattern
- Improved code documentation throughout

## [0.1.14] - 2025-11-27

### Added
- **DBSF (Distribution-Based Score Fusion)**: Z-score normalization with mean ± 3σ clipping
  - `dbsf` and `dbsf_multi` functions for 2+ lists
  - `FusionMethod::Dbsf` variant for unified API
  - More robust than min-max normalization when score distributions differ
- Updated prelude with `dbsf` export

## [0.1.13] - 2025-11-26

### Changed
- Added "Why This Library?" section to README with Bruch et al. (2022) citation
- Documented architectural benefits of dedicated fusion library

## [0.1.12] - 2025-11-26

### Changed
- Updated README with RRF k parameter tuning guide (Cormack 2009)
- Added k value recommendations by use case
- Documented `rrf_weighted` in algorithms table

## [0.1.11] - 2025-11-26

### Added
- `rrf_weighted` for per-retriever weighted rank fusion
- Tests for weighted RRF formula and error handling

## [0.1.10] - 2025-11-26

### Added
- Mutation-killing unit tests for RRF, weighted, combsum, combmnz
- Tests for exact score formulas and weight normalization

## [0.1.9] - 2025-11-26

### Changed
- Simplified module documentation
- Cleaner algorithm comparison table

## [0.1.8] - 2025-11-26

### Added
- **Improved README**: Complete e2e example, algorithm decision guide
- **12 integration tests**: Realistic e2e workflows:
  - `e2e_hybrid_bm25_dense`: Real-world hybrid search
  - `e2e_three_way_fusion`: 3+ retriever combination
  - `e2e_weighted_tuned`: Weight tuning verification
  - `e2e_buffer_reuse`: High-throughput pattern
  - `e2e_combmnz_overlap_bonus`: Overlap reward verification
  - `e2e_rrf_k_tuning`: k parameter effect
  - `e2e_empty_lists`: Edge case handling
  - `e2e_integer_ids`: Non-string ID support
  - `e2e_top_k_filtering`: Result limiting
  - `e2e_deterministic`: Consistent output verification
  - `e2e_borda_ordering`: Borda score correctness
  - `e2e_weighted_multi_errors`: Error handling

### Changed
- README now explains when/why to use each algorithm
- Added "When to Use Rank Fusion" section with score range table

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
