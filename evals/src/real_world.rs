//! Real-world dataset evaluation infrastructure.
//!
//! This module provides utilities for evaluating fusion methods on actual IR datasets
//! like MS MARCO, BEIR, or TREC runs.

use anyhow::{Context, Result};
use rank_eval::graded::{compute_ndcg, compute_map};
use rank_fusion::{
    additive_multi_task_with_config, borda, combmnz, combsum, dbsf, isr, rrf,
    standardized_with_config, weighted, AdditiveMultiTaskConfig, Normalization, StandardizedConfig,
    WeightedConfig,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

// Re-export types and functions from rank-eval for backward compatibility
#[allow(unused_imports)] // Re-exports for external use
pub use rank_eval::trec::{TrecRun, Qrel, load_trec_runs, load_qrels, group_runs_by_query, group_qrels_by_query};

/// Evaluation metrics for a fusion method.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FusionMetrics {
    pub ndcg_at_10: f64,
    pub ndcg_at_100: f64,
    pub map: f64,
    pub mrr: f64,
    pub precision_at_10: f64,
    pub recall_at_100: f64,
}

/// Fusion method configuration for evaluation.
#[derive(Debug, Clone)]
pub enum FusionMethod {
    Rrf { k: u32 },
    Isr { k: u32 },
    CombSum,
    CombMnz,
    Borda,
    Dbsf,
    Weighted { weight_a: f32, weight_b: f32, normalize: bool },
    Standardized { clip_range: (f32, f32) },
    AdditiveMultiTask { weight_a: f32, weight_b: f32, normalization: Normalization },
}

impl FusionMethod {
    /// Human-readable name for this fusion method.
    pub fn name(&self) -> String {
        match self {
            Self::Rrf { k } => format!("rrf_k{}", k),
            Self::Isr { k } => format!("isr_k{}", k),
            Self::CombSum => "combsum".to_string(),
            Self::CombMnz => "combmnz".to_string(),
            Self::Borda => "borda".to_string(),
            Self::Dbsf => "dbsf".to_string(),
            Self::Weighted { weight_a, weight_b, normalize } => {
                format!("weighted_{}_{}_norm{}", weight_a, weight_b, normalize)
            }
            Self::Standardized { clip_range } => {
                format!("standardized_{}_{}", clip_range.0, clip_range.1)
            }
            Self::AdditiveMultiTask { weight_a, weight_b, normalization } => {
                format!("additive_multi_task_{}_{}_{:?}", weight_a, weight_b, normalization)
            }
        }
    }

    /// Fuse two ranked lists using this method.
    #[allow(dead_code)] // Used in tests
    pub fn fuse(&self, a: &[(String, f32)], b: &[(String, f32)]) -> Vec<(String, f32)> {
        match self {
            Self::Rrf { k: _ } => rrf(a, b),
            Self::Isr { k: _ } => isr(a, b),
            Self::CombSum => combsum(a, b),
            Self::CombMnz => combmnz(a, b),
            Self::Borda => borda(a, b),
            Self::Dbsf => dbsf(a, b),
            Self::Weighted { weight_a, weight_b, normalize } => {
                weighted(a, b, WeightedConfig {
                    weight_a: *weight_a,
                    weight_b: *weight_b,
                    normalize: *normalize,
                    top_k: None,
                })
            }
            Self::Standardized { clip_range } => {
                standardized_with_config(a, b, StandardizedConfig::new(*clip_range))
            }
            Self::AdditiveMultiTask { weight_a, weight_b, normalization } => {
                additive_multi_task_with_config(
                    a,
                    b,
                    AdditiveMultiTaskConfig {
                        weights: (*weight_a, *weight_b),
                        normalization: *normalization,
                        top_k: None,
                    },
                )
            }
        }
    }
}

// TREC parsing functions are now imported from rank-eval crate

/// Evaluate a fusion method on grouped runs.
pub fn evaluate_fusion_method(
    grouped_runs: &HashMap<String, HashMap<String, Vec<(String, f32)>>>,
    qrels: &HashMap<String, HashMap<String, u32>>,
    method: &FusionMethod,
) -> FusionMetrics {
    let mut metrics = FusionMetrics {
        ndcg_at_10: 0.0,
        ndcg_at_100: 0.0,
        map: 0.0,
        mrr: 0.0,
        precision_at_10: 0.0,
        recall_at_100: 0.0,
    };

    let mut query_count = 0;

    let mut skipped_queries = 0;
    let mut skipped_reason = String::new();

    for (query_id, runs) in grouped_runs {
        // Need at least 2 runs to fuse
        if runs.len() < 2 {
            skipped_queries += 1;
            if skipped_queries == 1 {
                skipped_reason = format!("Query '{}' has only {} run(s), need at least 2", query_id, runs.len());
            }
            continue;
        }

        let run_vecs: Vec<&Vec<(String, f32)>> = runs.values().collect();
        // Redundant check removed - if runs.len() < 2, run_vecs.len() will also be < 2

        // Fuse all available runs (multi-run fusion)
        // Use the rank-fusion crate's fuse_multi which is more efficient and correct
        // Convert our wrapper FusionMethod to the crate's FusionMethod
        let crate_method = match method {
            FusionMethod::Rrf { k } => rank_fusion::FusionMethod::Rrf { k: *k },
            FusionMethod::Isr { k } => rank_fusion::FusionMethod::Isr { k: *k },
            FusionMethod::CombSum => rank_fusion::FusionMethod::CombSum,
            FusionMethod::CombMnz => rank_fusion::FusionMethod::CombMnz,
            FusionMethod::Borda => rank_fusion::FusionMethod::Borda,
            FusionMethod::Dbsf => rank_fusion::FusionMethod::Dbsf,
            FusionMethod::Weighted { weight_a, weight_b, normalize } => {
                rank_fusion::FusionMethod::Weighted {
                    weight_a: *weight_a,
                    weight_b: *weight_b,
                    normalize: *normalize,
                }
            },
            FusionMethod::Standardized { clip_range } => {
                rank_fusion::FusionMethod::Standardized { clip_range: *clip_range }
            },
            FusionMethod::AdditiveMultiTask { weight_a, weight_b, normalization } => {
                rank_fusion::FusionMethod::AdditiveMultiTask {
                    weight_a: *weight_a,
                    weight_b: *weight_b,
                    normalization: *normalization,
                }
            },
        };
        
        // Use fuse_multi for proper multi-run fusion
        let run_slices: Vec<&[(String, f32)]> = run_vecs.iter().map(|v| v.as_slice()).collect();
        let fused = crate_method.fuse_multi(&run_slices);
        
        if fused.is_empty() {
            skipped_queries += 1;
            if skipped_queries == 1 {
                skipped_reason = format!("Query '{}' produced empty fusion result", query_id);
            }
            continue;
        }

        if let Some(query_qrels) = qrels.get(query_id) {
            let query_metrics = compute_metrics(&fused, query_qrels);
            metrics.ndcg_at_10 += query_metrics.ndcg_at_10;
            metrics.ndcg_at_100 += query_metrics.ndcg_at_100;
            metrics.map += query_metrics.map;
            metrics.mrr += query_metrics.mrr;
            metrics.precision_at_10 += query_metrics.precision_at_10;
            metrics.recall_at_100 += query_metrics.recall_at_100;
            query_count += 1;
        }
    }

    if query_count > 0 {
        metrics.ndcg_at_10 /= query_count as f64;
        metrics.ndcg_at_100 /= query_count as f64;
        metrics.map /= query_count as f64;
        metrics.mrr /= query_count as f64;
        metrics.precision_at_10 /= query_count as f64;
        metrics.recall_at_100 /= query_count as f64;
    } else {
        // Log warning if no queries were evaluated
        eprintln!("Warning: No queries evaluated. Skipped {} queries. First reason: {}", 
            skipped_queries, 
            if skipped_queries > 0 { &skipped_reason } else { "No queries with qrels" }
        );
    }

    metrics
}

/// Evaluate all fusion methods on grouped runs.
///
/// This function is used by the evaluation runner to test all fusion methods
/// on a dataset.
#[allow(dead_code)] // Used in evaluate_real_world.rs
pub fn evaluate_all_methods(
    grouped_runs: &HashMap<String, HashMap<String, Vec<(String, f32)>>>,
    qrels: &HashMap<String, HashMap<String, u32>>,
) -> HashMap<String, FusionMetrics> {
    let methods = vec![
        FusionMethod::Rrf { k: 60 },
        FusionMethod::Isr { k: 1 },
        FusionMethod::CombSum,
        FusionMethod::CombMnz,
        FusionMethod::Borda,
        FusionMethod::Dbsf,
        FusionMethod::Weighted {
            weight_a: 0.7,
            weight_b: 0.3,
            normalize: true,
        },
        FusionMethod::Weighted {
            weight_a: 0.9,
            weight_b: 0.1,
            normalize: true,
        },
        FusionMethod::Standardized {
            clip_range: (-3.0, 3.0),
        },
        FusionMethod::Standardized {
            clip_range: (-2.0, 2.0),
        },
        FusionMethod::AdditiveMultiTask {
            weight_a: 1.0,
            weight_b: 1.0,
            normalization: Normalization::ZScore,
        },
        FusionMethod::AdditiveMultiTask {
            weight_a: 1.0,
            weight_b: 20.0,
            normalization: Normalization::ZScore,
        },
    ];

    let mut results = HashMap::new();
    for method in methods {
        let metrics = evaluate_fusion_method(grouped_runs, qrels, &method);
        results.insert(method.name(), metrics);
    }

    results
}

/// Compute IR metrics for a ranked list.
///
/// This function is used internally by `evaluate_fusion_method`.
#[allow(dead_code)] // Used internally in evaluate_fusion_method
fn compute_metrics(
    ranked: &[(String, f32)],
    qrels: &HashMap<String, u32>,
) -> FusionMetrics {
    let mut metrics = FusionMetrics {
        ndcg_at_10: 0.0,
        ndcg_at_100: 0.0,
        map: 0.0,
        mrr: 0.0,
        precision_at_10: 0.0,
        recall_at_100: 0.0,
    };

    let relevant_docs: Vec<&String> = qrels
        .iter()
        .filter(|(_, &rel)| rel > 0)
        .map(|(doc_id, _)| doc_id)
        .collect();

    if relevant_docs.is_empty() {
        return metrics;
    }

    // Precision@10
    let top_10: Vec<&String> = ranked.iter().take(10).map(|(id, _)| id).collect();
    let relevant_in_top_10 = top_10.iter().filter(|id| qrels.get(id.as_str()).unwrap_or(&0) > &0).count();
    metrics.precision_at_10 = relevant_in_top_10 as f64 / 10.0;

    // Recall@100
    let top_100: Vec<&String> = ranked.iter().take(100).map(|(id, _)| id).collect();
    let relevant_in_top_100 = top_100.iter().filter(|id| qrels.get(id.as_str()).unwrap_or(&0) > &0).count();
    metrics.recall_at_100 = relevant_in_top_100 as f64 / relevant_docs.len() as f64;

    // MRR
    for (rank, (doc_id, _)) in ranked.iter().enumerate() {
        if qrels.get(doc_id.as_str()).unwrap_or(&0) > &0 {
            metrics.mrr = 1.0 / (rank + 1) as f64;
            break;
        }
    }

    // nDCG@10 and nDCG@100
    metrics.ndcg_at_10 = compute_ndcg(ranked, qrels, 10);
    metrics.ndcg_at_100 = compute_ndcg(ranked, qrels, 100);

    // MAP
    metrics.map = compute_map(ranked, qrels);

    metrics
}

// Graded metrics functions are now imported from rank-eval crate

/// Dataset information for evaluation.
///
/// This type is reserved for future use when implementing dataset metadata
/// loading from JSON configuration files.
#[allow(dead_code)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetInfo {
    pub name: String,
    pub description: String,
    pub queries: usize,
    pub documents: usize,
    pub runs_path: Option<PathBuf>,
    pub qrels_path: Option<PathBuf>,
}

/// Load dataset information from a JSON file.
///
/// This function is reserved for future use when implementing dataset metadata
/// loading from JSON configuration files.
#[allow(dead_code)]
pub fn load_dataset_info(path: impl AsRef<Path>) -> Result<DatasetInfo> {
    let content = std::fs::read_to_string(path.as_ref())
        .with_context(|| format!("Failed to read dataset info: {:?}", path.as_ref()))?;
    let info: DatasetInfo = serde_json::from_str(&content)
        .context("Failed to parse dataset info JSON")?;
    Ok(info)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_trec_runs() {
        let content = "1 Q0 doc1 1 0.9 run1\n1 Q0 doc2 2 0.8 run1\n";
        let temp_dir = tempfile::TempDir::new().unwrap();
        let temp_file = temp_dir.path().join("test_runs.txt");
        std::fs::write(&temp_file, content).unwrap();
        let runs = load_trec_runs(&temp_file).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].query_id, "1");
        assert_eq!(runs[0].doc_id, "doc1");
        // temp_dir will be cleaned up automatically
    }

    #[test]
    fn test_compute_ndcg() {
        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
            ("doc3".to_string(), 0.7),
        ];
        let mut qrels = HashMap::new();
        qrels.insert("doc1".to_string(), 2);
        qrels.insert("doc2".to_string(), 1);
        qrels.insert("doc3".to_string(), 0);

        let ndcg = compute_ndcg(&ranked, &qrels, 3);
        assert!(ndcg > 0.0);
        assert!(ndcg <= 1.0);
    }

    #[test]
    fn test_fusion_methods() {
        let a = vec![("d1".to_string(), 0.9), ("d2".to_string(), 0.8)];
        let b = vec![("d2".to_string(), 0.95), ("d3".to_string(), 0.7)];

        let methods = vec![
            FusionMethod::Rrf { k: 60 },
            FusionMethod::CombSum,
            FusionMethod::Standardized {
                clip_range: (-3.0, 3.0),
            },
        ];

        for method in methods {
            let fused = method.fuse(&a, &b);
            assert!(!fused.is_empty());
    }
}
}
