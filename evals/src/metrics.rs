//! Standard IR evaluation metrics.
//!
//! This module re-exports binary relevance metrics from the `rank-eval` crate
//! and provides a `Metrics` struct for convenience.

pub use rank_eval::binary::{
    average_precision, mrr, ndcg_at_k, precision_at_k, recall_at_k,
};

use std::collections::HashSet;

/// All metrics for a single ranking.
///
/// This struct wraps the binary metrics from `rank-eval` for convenience.
#[derive(Debug, Clone, serde::Serialize)]
pub struct Metrics {
    pub precision_at_1: f64,
    pub precision_at_5: f64,
    pub precision_at_10: f64,
    pub recall_at_5: f64,
    pub recall_at_10: f64,
    pub mrr: f64,
    pub ndcg_at_5: f64,
    pub ndcg_at_10: f64,
    pub average_precision: f64,
}

impl Metrics {
    pub fn compute<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> Self {
        Self {
            precision_at_1: precision_at_k(ranked, relevant, 1),
            precision_at_5: precision_at_k(ranked, relevant, 5),
            precision_at_10: precision_at_k(ranked, relevant, 10),
            recall_at_5: recall_at_k(ranked, relevant, 5),
            recall_at_10: recall_at_k(ranked, relevant, 10),
            mrr: mrr(ranked, relevant),
            ndcg_at_5: ndcg_at_k(ranked, relevant, 5),
            ndcg_at_10: ndcg_at_k(ranked, relevant, 10),
            average_precision: average_precision(ranked, relevant),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_precision_at_k() {
        let ranked = vec!["a", "b", "c", "d", "e"];
        let relevant: HashSet<_> = ["a", "c", "e"].into_iter().collect();

        assert!((precision_at_k(&ranked, &relevant, 1) - 1.0).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 2) - 0.5).abs() < 1e-9);
        assert!((precision_at_k(&ranked, &relevant, 5) - 0.6).abs() < 1e-9);
    }

    #[test]
    fn test_mrr() {
        let ranked = vec!["a", "b", "c"];
        let relevant: HashSet<_> = ["b"].into_iter().collect();

        assert!((mrr(&ranked, &relevant) - 0.5).abs() < 1e-9);
    }

    #[test]
    fn test_ndcg() {
        let ranked = vec!["a", "b", "c", "d"];
        let relevant: HashSet<_> = ["a", "c"].into_iter().collect();

        // a at pos 0, c at pos 2
        let dcg = 1.0 / 2.0_f64.log2() + 1.0 / 4.0_f64.log2();
        // ideal: both at top
        let idcg = 1.0 / 2.0_f64.log2() + 1.0 / 3.0_f64.log2();

        assert!((ndcg_at_k(&ranked, &relevant, 4) - dcg / idcg).abs() < 1e-9);
    }
}
