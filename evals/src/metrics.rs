//! Standard IR evaluation metrics.
//!
//! All metrics assume:
//! - `ranked`: List of document IDs in ranked order (best first)
//! - `relevant`: Set of document IDs that are relevant (ground truth)

use std::collections::HashSet;

/// Precision at k: fraction of top-k that are relevant.
///
/// P@k = |relevant ∩ top-k| / k
pub fn precision_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    if k == 0 {
        return 0.0;
    }
    let hits = ranked.iter().take(k).filter(|id| relevant.contains(id)).count();
    hits as f64 / k as f64
}

/// Recall at k: fraction of relevant docs in top-k.
///
/// R@k = |relevant ∩ top-k| / |relevant|
pub fn recall_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    let hits = ranked.iter().take(k).filter(|id| relevant.contains(id)).count();
    hits as f64 / relevant.len() as f64
}

/// Mean Reciprocal Rank: 1 / rank of first relevant document.
///
/// MRR = 1 / (rank of first relevant doc)
/// Returns 0.0 if no relevant docs found.
pub fn mrr<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            return 1.0 / (i + 1) as f64;
        }
    }
    0.0
}

/// Discounted Cumulative Gain at k.
///
/// DCG@k = Σᵢ (rel(i) / log₂(i + 2))
///
/// Uses binary relevance: rel(i) = 1 if relevant, 0 otherwise.
pub fn dcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    ranked
        .iter()
        .take(k)
        .enumerate()
        .filter(|(_, id)| relevant.contains(id))
        .map(|(i, _)| 1.0 / (i as f64 + 2.0).log2())
        .sum()
}

/// Ideal DCG at k (all relevant docs at top).
pub fn idcg_at_k(n_relevant: usize, k: usize) -> f64 {
    (0..k.min(n_relevant))
        .map(|i| 1.0 / (i as f64 + 2.0).log2())
        .sum()
}

/// Normalized DCG at k.
///
/// nDCG@k = DCG@k / IDCG@k
pub fn ndcg_at_k<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>, k: usize) -> f64 {
    let ideal = idcg_at_k(relevant.len(), k);
    if ideal == 0.0 {
        return 0.0;
    }
    dcg_at_k(ranked, relevant, k) / ideal
}

/// Average Precision: average of precision at each relevant doc.
///
/// AP = (1/|R|) × Σᵢ (P@i × rel(i))
pub fn average_precision<I: Eq + std::hash::Hash>(ranked: &[I], relevant: &HashSet<I>) -> f64 {
    if relevant.is_empty() {
        return 0.0;
    }
    
    let mut sum = 0.0;
    let mut hits = 0;
    
    for (i, id) in ranked.iter().enumerate() {
        if relevant.contains(id) {
            hits += 1;
            sum += hits as f64 / (i + 1) as f64;
        }
    }
    
    sum / relevant.len() as f64
}

/// All metrics for a single ranking.
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

