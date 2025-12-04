# Research Summary: Implementation Plans

## Overview

This document summarizes the research-backed improvements planned for `rank-fusion` and `rank-refine`, based on recent papers (2024-2025) with validated empirical results.

## Key Papers

1. **ERANK** (arXiv:2509.00520): Fine-grained scoring and standardization-based fusion
2. **ResFlow** (arXiv:2411.09705): Additive multi-task fusion
3. **TS-SetRank** (arXiv:2511.01208): Contextual relevance estimation

## rank-fusion Improvements

### 1. Standardization-Based Fusion (ERANK)

**Research Finding**: Z-score normalization (standardization) outperforms min-max normalization for hybrid BM25 + reranker fusion.

**Implementation**:
- Use `zscore_params()` helper (already exists for DBSF)
- Clip z-scores to [-3, 3] to prevent outliers
- Fallback to min-max for zero-variance cases

**Expected Impact**: 2-5% NDCG improvement when score distributions differ significantly.

**ERANK's Hybrid Formula**:
```
final_score = 0.2 × standardized_BM25 + 0.8 × standardized_rerank
```

### 2. Additive Multi-Task Fusion (ResFlow)

**Research Finding**: Additive fusion `α·CTR + β·CTCVR` consistently outperforms multiplicative `CTR^α × CVR^β` for e-commerce ranking.

**Implementation**:
- Support z-score or min-max normalization
- Weighted additive combination
- More robust to extreme values than multiplicative

**Expected Impact**: 1.29% OPU increase (validated in Shopee production A/B tests).

**ResFlow's Optimal Formula** (from their experiments):
```
Score = CTR + CTCVR × 20
```

### 3. QPP-Based Routing

**Research Finding**: Hard queries (low overlap, high variance) benefit from different fusion strategies than easy queries.

**Implementation**:
- Estimate query difficulty from result list characteristics
- Route easy queries to CombSUM, hard queries to RRF(k=20)
- Difficulty based on overlap ratio and coefficient of variation

**Expected Impact**: Improves quality for hard queries without degrading easy query performance.

## rank-refine Improvements

### 1. Fine-Grained Scoring (ERANK)

**Research Finding**: Integer scoring (0-10) with probability weighting significantly improves discrimination over binary classification.

**ERANK Formula**:
```
final_score = s_i × Pr(token = s_i)
```
where `s_i` is integer score (0-10) and `Pr(token = s_i)` is token probability.

**Implementation**:
- Map f32 similarity to u8 (0-10) using linear/quantile/custom mappers
- Support probability weighting (requires token logits from LLM)
- For embedding-based reranking, use similarity as confidence proxy

**Expected Impact**: 3-7% nDCG@10 improvement on BRIGHT, TREC DL, BEIR.

**ERANK Results**:
- Binary (yes/no): 20.8 nDCG@10 on BRIGHT
- 0-3 scale: 22.7 nDCG@10
- 0-10 scale: 23.2 nDCG@10

### 2. Contextual Relevance (TS-SetRank)

**Research Finding**: Document relevance is context-dependent—varies by batch composition and ordering, especially for reasoning-intensive queries.

**TS-SetRank Algorithm**:
- Phase I: Uniform exploration (25 rounds)
- Phase II: Thompson sampling (75 rounds)
- Beta-Bernoulli posteriors for uncertainty estimation
- Marginalize over all possible batches

**Expected Impact**: 15-25% nDCG@10 improvement on BRIGHT, 6-21% on BEIR.

**Variance Decomposition** (from TS-SetRank paper):
- Intrinsic (token sampling): ~55% of total variance
- Positional (document order): ~36% of total variance
- Compositional (batch contents): ~9% of total variance

### 3. Reasoning Explanations

**Research Finding**: Chain-of-thought reasoning traces improve interpretability and user trust.

**Implementation**:
- Generate reasoning steps for each ranking decision
- Token-level alignment information (for MaxSim)
- Confidence scores based on context variance

**Expected Impact**: UX enhancement, debugging aid (no quantitative improvement reported).

## Implementation Priority

### rank-fusion
1. Standardized fusion (highest impact, reuses existing code)
2. Additive multi-task fusion (validated in production)
3. QPP routing (requires more validation)

### rank-refine
1. Fine-grained scoring (highest impact, straightforward)
2. Contextual relevance (high impact, requires Bayesian inference)
3. Reasoning explanations (UX enhancement, lower priority)

## Validation Strategy

For each improvement:
1. Unit tests for edge cases (empty lists, zero variance, etc.)
2. Property tests (monotonicity, idempotency)
3. Benchmark against existing methods on real datasets
4. Integration tests with full pipeline

## Research Citations

- ERANK: Yuzheng Cai et al. "ERank: Fusing Supervised Fine-Tuning and Reinforcement Learning for Effective and Efficient Text Reranking" (arXiv:2509.00520, 2025)
- ResFlow: Cong Fu et al. "Residual Multi-Task Learner for Applied Ranking" (arXiv:2411.09705, 2024)
- TS-SetRank: Jerry Huang et al. "Contextual Relevance and Adaptive Sampling for LLM-Based Document Reranking" (arXiv:2511.01208, 2025)

