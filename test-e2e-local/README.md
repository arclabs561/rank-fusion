# E2E Validation Tests

This directory contains end-to-end validation tests that simulate how the published crates would be used by real consumers.

## Purpose

These tests verify that:
1. The crates work correctly when used as dependencies (not path dependencies)
2. The public APIs are correct and usable
3. Integration between crates (rank-fusion + rank-eval, etc.) works
4. Common use cases are supported

## Test Binaries

### `test-fusion-basic`
Tests basic rank-fusion functionality:
- All fusion algorithms (RRF, ISR, CombSUM, CombMNZ, Borda, DBSF, Weighted, Standardized, Additive Multi-Task)
- Multi-run fusion
- Configuration options

### `test-fusion-eval-integration`
Tests integration between rank-fusion and rank-eval:
- Fuse results from multiple retrievers
- Evaluate with binary metrics (nDCG, Precision, Recall, MRR)
- Evaluate with graded metrics (nDCG, MAP)

### `test-refine-basic`
Tests basic rank-refine functionality:
- MaxSim scoring
- ColBERT ranking
- Token pooling
- Cosine similarity

### `test-eval-basic`
Tests basic rank-eval functionality:
- Binary metrics (nDCG, Precision, Recall, MRR, AP)
- Graded metrics (nDCG, MAP)
- TREC format parsing

### `test-full-pipeline`
Tests a complete RAG pipeline:
1. Multiple retrievers (BM25, dense)
2. Fuse results
3. Refine with ColBERT
4. Evaluate with rank-eval

## Running Tests

```bash
# Run all tests
cargo run -p test-e2e-local --bin test-fusion-basic
cargo run -p test-e2e-local --bin test-fusion-eval-integration
cargo run -p test-e2e-local --bin test-refine-basic
cargo run -p test-e2e-local --bin test-eval-basic
cargo run -p test-e2e-local --bin test-full-pipeline

# Or run all at once
for bin in test-fusion-basic test-fusion-eval-integration test-refine-basic test-eval-basic test-full-pipeline; do
    cargo run -p test-e2e-local --bin $bin
done
```

## CI Integration

These tests can be run in CI to verify published packages work correctly. They use path dependencies that simulate published versions, ensuring the public API is correct.

## Future Enhancements

- Test with actual published versions from crates.io
- Test Python bindings
- Test WASM bindings
- Test with real datasets
- Performance benchmarks

