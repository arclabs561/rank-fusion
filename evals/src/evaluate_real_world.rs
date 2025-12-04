//! Comprehensive evaluation runner for real-world datasets.
//!
//! Evaluates all fusion methods on MS MARCO, BEIR, TREC, and other datasets.

use rank_eval::dataset::{get_dataset_stats, list_datasets, validate_dataset_dir, compute_comprehensive_stats, ComprehensiveStats, validate_dataset, DatasetValidationResult, DatasetStats};
use crate::dataset_registry::{DatasetEntry, DatasetRegistry};
use crate::real_world::{
    evaluate_all_methods, group_qrels_by_query, group_runs_by_query, FusionMetrics,
};
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;

/// Evaluation result for a single dataset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEvaluationResult {
    pub dataset_name: String,
    pub dataset_stats: DatasetStats,
    pub comprehensive_stats: Option<ComprehensiveStats>,
    pub validation_result: Option<DatasetValidationResult>,
    pub dataset_metadata: Option<DatasetEntry>,
    pub method_results: HashMap<String, FusionMetrics>,
    pub best_method: String,
    pub best_ndcg_at_10: f64,
}

/// Comprehensive evaluation results across all datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationResults {
    pub datasets: Vec<DatasetEvaluationResult>,
    pub summary: EvaluationSummary,
}

/// Summary statistics across all datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvaluationSummary {
    pub total_datasets: usize,
    pub total_queries: usize,
    pub method_averages: HashMap<String, MethodAverageMetrics>,
    pub best_methods_per_dataset: HashMap<String, String>,
}

/// Average metrics for a fusion method across all datasets.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MethodAverageMetrics {
    pub avg_ndcg_at_10: f64,
    pub avg_ndcg_at_100: f64,
    pub avg_map: f64,
    pub avg_mrr: f64,
    pub avg_precision_at_10: f64,
    pub avg_recall_at_100: f64,
}

/// Normalize dataset name for registry lookup.
/// Handles variations like "msmarco" vs "msmarco-passage", directory names, etc.
fn normalize_dataset_name(name: &str) -> String {
    let normalized = name
        .to_lowercase()
        .replace("_", "-")
        .replace(" ", "-");
    
    // Try direct match first
    let registry = DatasetRegistry::new();
    if registry.get(&normalized).is_some() {
        return normalized;
    }
    
    // Try common variations
    let variations = vec![
        normalized.clone(),
        normalized.replace("-passage", ""),
        normalized.replace("-document", ""),
        normalized.replace("trec-", ""),
        normalized.replace("trec", ""),
    ];
    
    for variant in variations {
        if registry.get(&variant).is_some() {
            return variant;
        }
    }
    
    // Return normalized version even if not found in registry
    normalized
}

/// Evaluate a single dataset.
pub fn evaluate_dataset(
    dataset_name: &str,
    runs: &[crate::real_world::TrecRun],
    qrels: &[crate::real_world::Qrel],
) -> Result<DatasetEvaluationResult> {
    let stats = get_dataset_stats(runs, qrels);
    
    // Compute comprehensive statistics
    let comprehensive_stats = compute_comprehensive_stats(runs, qrels);

    let grouped_runs = group_runs_by_query(runs);
    let grouped_qrels = group_qrels_by_query(qrels);

    let method_results = evaluate_all_methods(&grouped_runs, &grouped_qrels);

    // Find best method by nDCG@10
    let (best_method, best_ndcg) = method_results
        .iter()
        .map(|(name, metrics)| (name.clone(), metrics.ndcg_at_10))
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .unwrap_or_else(|| ("none".to_string(), 0.0));

    // Look up dataset metadata from registry
    let normalized_name = normalize_dataset_name(dataset_name);
    let registry = DatasetRegistry::new();
    let dataset_metadata = registry.get(&normalized_name).cloned();

    Ok(DatasetEvaluationResult {
        dataset_name: dataset_name.to_string(),
        dataset_stats: stats,
        comprehensive_stats: Some(comprehensive_stats),
        validation_result: None, // Will be set by caller if validation was performed
        dataset_metadata,
        method_results,
        best_method,
        best_ndcg_at_10: best_ndcg,
    })
}

/// Evaluate multiple datasets from a directory.
pub fn evaluate_datasets_dir(datasets_dir: impl AsRef<Path>) -> Result<EvaluationResults> {
    let datasets = list_datasets(&datasets_dir)?;
    let mut results = Vec::new();

    for dataset_name in &datasets {
        let dataset_path = datasets_dir.as_ref().join(dataset_name);
        
        if !validate_dataset_dir(&dataset_path)? {
            eprintln!("Skipping invalid dataset: {}", dataset_name);
            continue;
        }

        // Find run files (exclude qrels files)
        let run_files: Vec<String> = std::fs::read_dir(&dataset_path)
            .context("Failed to read dataset directory")?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.path().file_name()?.to_str()?.to_string();
                // Include .run and .txt files, but exclude qrels files
                if (name.ends_with(".run") || name.ends_with(".txt")) 
                    && !name.contains("qrels") 
                    && name != "qrels.txt" 
                    && name != "qrels" {
                    Some(name)
                } else {
                    None
                }
            })
            .collect();

        if run_files.is_empty() {
            eprintln!("No run files found for dataset: {}", dataset_name);
            continue;
        }

        // Load runs
        let run_paths: Vec<&str> = run_files.iter().map(|s| s.as_str()).collect();
        let runs = rank_eval::dataset::load_trec_runs_from_dir(&dataset_path, &run_paths)
            .with_context(|| format!("Failed to load runs for dataset: {}", dataset_name))?;

        // Load qrels
        let qrels = rank_eval::dataset::load_trec_qrels_from_dir(&dataset_path)
            .with_context(|| format!("Failed to load qrels for dataset: {}", dataset_name))?;

        if runs.is_empty() || qrels.is_empty() {
            eprintln!("Empty runs or qrels for dataset: {}", dataset_name);
            continue;
        }

        // Validate dataset before evaluation (optional, don't fail on validation errors)
        let qrels_path = dataset_path.join("qrels.txt");
        let validation_result = validate_dataset(
            &dataset_path.join(&run_files[0]),
            &qrels_path
        ).ok();
        
        if let Some(ref validation) = validation_result {
            if !validation.is_valid && !validation.errors.is_empty() {
                eprintln!("  Warning: Dataset validation issues found:");
                for error in &validation.errors {
                    eprintln!("    - {}", error);
                }
            }

            if !validation.warnings.is_empty() {
                for warning in &validation.warnings {
                    eprintln!("  Warning: {}", warning);
                }
            }

            println!("Evaluating dataset: {} ({} runs, {} qrels, {} queries ready for fusion)", 
                dataset_name, 
                runs.len(), 
                qrels.len(),
                validation.statistics.queries_in_both);
        } else {
            println!("Evaluating dataset: {} ({} runs, {} qrels)", 
                dataset_name, runs.len(), qrels.len());
        }

        match evaluate_dataset(dataset_name, &runs, &qrels) {
            Ok(mut result) => {
                // Attach validation result if available
                result.validation_result = validation_result;
                println!("  Best method: {} (nDCG@10: {:.4})", result.best_method, result.best_ndcg_at_10);
                results.push(result);
            }
            Err(e) => {
                eprintln!("  Error evaluating dataset {}: {}", dataset_name, e);
            }
        }
    }

    let summary = compute_summary(&results);

    Ok(EvaluationResults {
        datasets: results,
        summary,
    })
}

/// Compute summary statistics across all datasets.
fn compute_summary(results: &[DatasetEvaluationResult]) -> EvaluationSummary {
    let total_datasets = results.len();
    let total_queries: usize = results.iter().map(|r| r.dataset_stats.unique_queries).sum();

    // Collect all method names
    let all_methods: std::collections::HashSet<String> = results
        .iter()
        .flat_map(|r| r.method_results.keys())
        .cloned()
        .collect();

    // Compute averages per method
    let mut method_averages = HashMap::new();
    for method_name in &all_methods {
        let mut sum_ndcg_10 = 0.0;
        let mut sum_ndcg_100 = 0.0;
        let mut sum_map = 0.0;
        let mut sum_mrr = 0.0;
        let mut sum_p10 = 0.0;
        let mut sum_r100 = 0.0;
        let mut count = 0;

        for result in results {
            if let Some(metrics) = result.method_results.get(method_name) {
                sum_ndcg_10 += metrics.ndcg_at_10;
                sum_ndcg_100 += metrics.ndcg_at_100;
                sum_map += metrics.map;
                sum_mrr += metrics.mrr;
                sum_p10 += metrics.precision_at_10;
                sum_r100 += metrics.recall_at_100;
                count += 1;
            }
        }

        if count > 0 {
            method_averages.insert(
                method_name.clone(),
                MethodAverageMetrics {
                    avg_ndcg_at_10: sum_ndcg_10 / count as f64,
                    avg_ndcg_at_100: sum_ndcg_100 / count as f64,
                    avg_map: sum_map / count as f64,
                    avg_mrr: sum_mrr / count as f64,
                    avg_precision_at_10: sum_p10 / count as f64,
                    avg_recall_at_100: sum_r100 / count as f64,
                },
            );
        }
    }

    // Best methods per dataset
    let best_methods_per_dataset: HashMap<String, String> = results
        .iter()
        .map(|r| (r.dataset_name.clone(), r.best_method.clone()))
        .collect();

    EvaluationSummary {
        total_datasets,
        total_queries,
        method_averages,
        best_methods_per_dataset,
    }
}

/// Generate HTML report for evaluation results.
pub fn generate_html_report(results: &EvaluationResults, output_path: impl AsRef<Path>) -> Result<()> {
    let html = format!(
        r#"<!DOCTYPE html>
<html>
<head>
    <title>Rank Fusion Real-World Evaluation Report</title>
    <style>
        body {{ 
            font-family: 'SF Mono', 'Menlo', monospace; 
            max-width: 1600px; 
            margin: 0 auto; 
            padding: 20px;
            background: #1a1a2e;
            color: #eee;
        }}
        h1 {{ color: #00d9ff; border-bottom: 2px solid #00d9ff; padding-bottom: 10px; }}
        h2 {{ color: #ff6b9d; margin-top: 40px; }}
        .summary {{ 
            background: #0f3460; 
            padding: 20px; 
            border-radius: 8px; 
            margin-bottom: 30px;
        }}
        .stat {{ display: inline-block; margin-right: 30px; }}
        .stat-value {{ font-size: 2em; color: #00ff88; }}
        .stat-label {{ color: #888; }}
        .dataset {{ 
            background: #16213e; 
            padding: 20px; 
            margin: 20px 0; 
            border-radius: 8px;
            border-left: 4px solid #00d9ff;
        }}
        .metadata {{
            background: #0f3460;
            padding: 15px;
            border-radius: 6px;
            margin: 15px 0;
            border-left: 3px solid #ffd93d;
        }}
        .metadata p {{
            margin: 8px 0;
            font-size: 14px;
        }}
        .metadata a {{
            color: #00d9ff;
            text-decoration: none;
        }}
        .metadata a:hover {{
            text-decoration: underline;
        }}
        table {{ 
            border-collapse: collapse; 
            width: 100%; 
            margin: 15px 0;
            font-size: 13px;
        }}
        th, td {{ 
            border: 1px solid #333; 
            padding: 8px 12px; 
            text-align: left; 
        }}
        th {{ 
            background: #0f3460; 
            color: #00d9ff;
        }}
        tr:nth-child(even) {{ background: #1a1a2e; }}
        tr:hover {{ background: #0f3460; }}
        .winner {{ background: #00ff88 !important; color: #000; font-weight: bold; }}
        .metric-best {{ color: #00ff88; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Rank Fusion Real-World Evaluation Report</h1>
    
    <div class="summary">
        <div class="stat">
            <div class="stat-value">{}</div>
            <div class="stat-label">Datasets Evaluated</div>
        </div>
        <div class="stat">
            <div class="stat-value">{}</div>
            <div class="stat-label">Total Queries</div>
        </div>
    </div>

    <h2>Method Averages Across All Datasets</h2>
    <table>
        <tr>
            <th>Method</th>
            <th>Avg nDCG@10</th>
            <th>Avg nDCG@100</th>
            <th>Avg MAP</th>
            <th>Avg MRR</th>
            <th>Avg P@10</th>
            <th>Avg R@100</th>
        </tr>
"#,
        results.summary.total_datasets, results.summary.total_queries
    );

    // Add dataset-specific results
    let mut html_body = html.to_string();
    
    // Sort methods by average nDCG@10
    let mut method_avgs: Vec<_> = results.summary.method_averages.iter().collect();
    method_avgs.sort_by(|(_, a), (_, b)| {
        b.avg_ndcg_at_10
            .partial_cmp(&a.avg_ndcg_at_10)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let best_ndcg = method_avgs.first().map(|(_, m)| m.avg_ndcg_at_10).unwrap_or(0.0);

    for (method_name, metrics) in &method_avgs {
        let is_best = (metrics.avg_ndcg_at_10 - best_ndcg).abs() < 1e-9;
        let row_class = if is_best { "winner" } else { "" };
        let ndcg_class = if is_best { "metric-best" } else { "" };

        html_body.push_str(&format!(
            r#"        <tr class="{}">
            <td>{}</td>
            <td class="{}">{:.4}</td>
            <td>{:.4}</td>
            <td>{:.4}</td>
            <td>{:.4}</td>
            <td>{:.4}</td>
            <td>{:.4}</td>
        </tr>
"#,
            row_class,
            method_name,
            ndcg_class,
            metrics.avg_ndcg_at_10,
            metrics.avg_ndcg_at_100,
            metrics.avg_map,
            metrics.avg_mrr,
            metrics.avg_precision_at_10,
            metrics.avg_recall_at_100
        ));
    }
    html_body.push_str("    </table>\n\n    <h2>Dataset-Specific Results</h2>\n");

    for dataset_result in &results.datasets {
        let best_ndcg = dataset_result
            .method_results
            .values()
            .map(|m| m.ndcg_at_10)
            .fold(0.0, f64::max);

        html_body.push_str(&format!(
            r#"
    <div class="dataset">
        <h3>{}</h3>
"#,
            dataset_result.dataset_name
        ));

        // Add dataset metadata if available
        if let Some(metadata) = &dataset_result.dataset_metadata {
            html_body.push_str(&format!(
                r#"        <div class="metadata">
            <p><strong>Description:</strong> {}</p>
            <p><strong>Category:</strong> {:?} | <strong>Priority:</strong> {} | <strong>Domain:</strong> {}</p>
"#,
                metadata.description,
                metadata.category,
                metadata.priority,
                metadata.domain.as_ref().unwrap_or(&"General".to_string())
            ));
            
            if let Some(notes) = &metadata.notes {
                html_body.push_str(&format!(r#"            <p><strong>Notes:</strong> {}</p>"#, notes));
            }
            
            if let Some(url) = &metadata.url {
                html_body.push_str(&format!(r#"            <p><strong>URL:</strong> <a href="{}" target="_blank">{}</a></p>"#, url, url));
            }
            
            html_body.push_str("        </div>\n");
        }

        html_body.push_str(&format!(
            r#"        <p>Queries: {} | Documents: {} | Runs: {} | Qrels: {}</p>
        <p>Best Method: <strong>{}</strong> (nDCG@10: {:.4})</p>
        <table>
            <tr>
                <th>Method</th>
                <th>nDCG@10</th>
                <th>nDCG@100</th>
                <th>MAP</th>
                <th>MRR</th>
                <th>P@10</th>
                <th>R@100</th>
            </tr>
"#,
            dataset_result.dataset_stats.unique_queries,
            dataset_result.dataset_stats.unique_documents,
            dataset_result.dataset_stats.unique_run_tags,
            dataset_result.dataset_stats.total_qrels,
            dataset_result.best_method,
            dataset_result.best_ndcg_at_10
        ));

        // Sort methods by nDCG@10
        let mut methods: Vec<_> = dataset_result.method_results.iter().collect();
        methods.sort_by(|(_, a), (_, b)| {
            b.ndcg_at_10
                .partial_cmp(&a.ndcg_at_10)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for (method_name, metrics) in &methods {
            let is_winner = method_name.as_str() == dataset_result.best_method.as_str();
            let row_class = if is_winner { "winner" } else { "" };
            let ndcg_class = if (metrics.ndcg_at_10 - best_ndcg).abs() < 1e-9 {
                "metric-best"
            } else {
                ""
            };

            html_body.push_str(&format!(
                r#"            <tr class="{}">
                <td>{}</td>
                <td class="{}">{:.4}</td>
                <td>{:.4}</td>
                <td>{:.4}</td>
                <td>{:.4}</td>
                <td>{:.4}</td>
                <td>{:.4}</td>
            </tr>
"#,
                row_class,
                method_name,
                ndcg_class,
                metrics.ndcg_at_10,
                metrics.ndcg_at_100,
                metrics.map,
                metrics.mrr,
                metrics.precision_at_10,
                metrics.recall_at_100
            ));
        }

        html_body.push_str("        </table>\n");
        
        // Add comprehensive statistics section if available
        if let Some(ref comp_stats) = dataset_result.comprehensive_stats {
            html_body.push_str(&format!(
                r#"
        <h4>Dataset Statistics</h4>
        <div style="margin: 15px 0;">
            <p><strong>Score Distribution:</strong> Min: {:.4}, Max: {:.4}, Mean: {:.4}, Median: {:.4}, Std Dev: {:.4}</p>
            <p><strong>Quality Metrics:</strong> Fusion Readiness: {:.1}%, Avg Runs per Query: {:.2}</p>
            <p><strong>Overlap:</strong> Query Overlap: {:.1}%, Document Overlap: {:.1}%</p>
        </div>
"#,
                comp_stats.runs.score_distribution.min,
                comp_stats.runs.score_distribution.max,
                comp_stats.runs.score_distribution.mean,
                comp_stats.runs.score_distribution.median,
                comp_stats.runs.score_distribution.std_dev,
                comp_stats.quality.fusion_readiness_ratio * 100.0,
                comp_stats.quality.avg_runs_per_query,
                comp_stats.overlap.query_overlap_ratio * 100.0,
                comp_stats.overlap.document_overlap_ratio * 100.0,
            ));
        }
        
        // Add validation results section if available
        if let Some(ref validation) = dataset_result.validation_result {
            if !validation.is_valid || !validation.warnings.is_empty() {
                html_body.push_str(&format!(
                    r#"
        <h4>Validation Results</h4>
        <div style="margin: 15px 0; padding: 10px; background: {}; border-radius: 4px;">
            <p><strong>Status:</strong> {}</p>
"#,
                    if validation.is_valid { "#0f3460" } else { "#4a1a1a" },
                    if validation.is_valid { "✓ Valid" } else { "✗ Invalid" }
                ));
                
                if !validation.errors.is_empty() {
                    html_body.push_str("<p><strong>Errors:</strong></p><ul>");
                    for error in &validation.errors {
                        html_body.push_str(&format!("<li>{}</li>", error));
                    }
                    html_body.push_str("</ul>");
                }
                
                if !validation.warnings.is_empty() {
                    html_body.push_str("<p><strong>Warnings:</strong></p><ul>");
                    for warning in &validation.warnings {
                        html_body.push_str(&format!("<li>{}</li>", warning));
                    }
                    html_body.push_str("</ul>");
                }
                
                html_body.push_str("</div>");
            }
        }
        
        html_body.push_str("    </div>\n");
    }

    html_body.push_str("</body>\n</html>");

    std::fs::write(output_path.as_ref(), html_body)
        .with_context(|| format!("Failed to write HTML report: {:?}", output_path.as_ref()))?;

    Ok(())
}

