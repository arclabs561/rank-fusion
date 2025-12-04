//! Binary for evaluating fusion methods on real-world datasets.
//!
//! Usage:
//!   cargo run --bin evaluate-real-world -- --datasets-dir ./datasets
//!   cargo run --bin evaluate-real-world -- --dataset ./datasets/msmarco

use anyhow::{Context, Result};
use clap::Parser;
use std::path::PathBuf;

// Import from parent crate modules  
use rank_fusion_evals::evaluate_real_world;

#[derive(Parser)]
#[command(name = "evaluate-real-world")]
#[command(about = "Evaluate rank fusion methods on real-world datasets")]
struct Args {
    /// Directory containing multiple datasets
    #[arg(long)]
    datasets_dir: Option<PathBuf>,

    /// Single dataset directory to evaluate
    #[arg(long)]
    dataset: Option<PathBuf>,

    /// Output path for HTML report
    #[arg(long, default_value = "real_world_eval_report.html")]
    output: PathBuf,

    /// Output path for JSON results
    #[arg(long, default_value = "real_world_eval_results.json")]
    json_output: PathBuf,
}

fn main() -> Result<()> {
    let args = Args::parse();

    let results = if let Some(datasets_dir) = args.datasets_dir {
        println!("Evaluating all datasets in: {:?}", datasets_dir);
        evaluate_real_world::evaluate_datasets_dir(&datasets_dir)?
    } else if let Some(dataset) = args.dataset {
        println!("Evaluating single dataset: {:?}", dataset);
        
        if !dataset.is_dir() {
            anyhow::bail!("Dataset path must be a directory: {:?}", dataset);
        }
        
        // Validate dataset directory
        if !rank_eval::dataset::validate_dataset_dir(&dataset)? {
            anyhow::bail!("Invalid dataset directory: {:?}. Need run files and qrels.txt", dataset);
        }
        
        // Find run files
        let run_files: Vec<String> = std::fs::read_dir(&dataset)
            .context("Failed to read dataset directory")?
            .filter_map(|e| e.ok())
            .filter_map(|e| {
                let name = e.path().file_name()?.to_str()?.to_string();
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
            anyhow::bail!("No run files found in dataset directory: {:?}", dataset);
        }
        
        // Load runs and qrels
        let run_paths: Vec<&str> = run_files.iter().map(|s| s.as_str()).collect();
        let runs = rank_eval::dataset::load_trec_runs_from_dir(&dataset, &run_paths)
            .context("Failed to load runs")?;
        let qrels = rank_eval::dataset::load_trec_qrels_from_dir(&dataset)
            .context("Failed to load qrels")?;
        
        if runs.is_empty() || qrels.is_empty() {
            anyhow::bail!("Empty runs or qrels in dataset: {:?}", dataset);
        }
        
        // Evaluate single dataset
        let dataset_name = dataset.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("dataset")
            .to_string();
        
        let result = evaluate_real_world::evaluate_dataset(&dataset_name, &runs, &qrels)
            .context("Failed to evaluate dataset")?;
        
        // Create results structure for single dataset
        let summary = evaluate_real_world::EvaluationSummary {
            total_datasets: 1,
            total_queries: result.dataset_stats.unique_queries,
            method_averages: result.method_results.iter()
                .map(|(name, metrics)| {
                    (name.clone(), evaluate_real_world::MethodAverageMetrics {
                        avg_ndcg_at_10: metrics.ndcg_at_10,
                        avg_ndcg_at_100: metrics.ndcg_at_100,
                        avg_map: metrics.map,
                        avg_mrr: metrics.mrr,
                        avg_precision_at_10: metrics.precision_at_10,
                        avg_recall_at_100: metrics.recall_at_100,
                    })
                })
                .collect(),
            best_methods_per_dataset: {
                let mut map = std::collections::HashMap::new();
                map.insert(dataset_name.clone(), result.best_method.clone());
                map
            },
        };
        
        let results = evaluate_real_world::EvaluationResults {
            datasets: vec![result],
            summary,
        };
        
        // Generate HTML report
        println!("Generating HTML report...");
        evaluate_real_world::generate_html_report(&results, &args.output)
            .with_context(|| format!("Failed to generate HTML report: {:?}", args.output))?;
        println!("HTML report written to: {:?}", args.output);

        // Write JSON results
        let json = serde_json::to_string_pretty(&results)?;
        std::fs::write(&args.json_output, json)
            .with_context(|| format!("Failed to write JSON results: {:?}", args.json_output))?;
        println!("JSON results written to: {:?}", args.json_output);

        // Print summary
        println!("\n=== Evaluation Summary ===");
        println!("Dataset: {}", dataset_name);
        println!("Queries: {}", results.summary.total_queries);
        println!("Best method: {}", results.summary.best_methods_per_dataset.get(&dataset_name).unwrap_or(&"none".to_string()));
        
        return Ok(());
    } else {
        anyhow::bail!("Must specify either --datasets-dir or --dataset");
    };

    // Generate HTML report
    println!("Generating HTML report...");
    evaluate_real_world::generate_html_report(&results, &args.output)
        .with_context(|| format!("Failed to generate HTML report: {:?}", args.output))?;
    println!("HTML report written to: {:?}", args.output);

    // Write JSON results
    let json = serde_json::to_string_pretty(&results)?;
    std::fs::write(&args.json_output, json)
        .with_context(|| format!("Failed to write JSON results: {:?}", args.json_output))?;
    println!("JSON results written to: {:?}", args.json_output);

    // Print summary
    println!("\n=== Evaluation Summary ===");
    println!("Datasets evaluated: {}", results.summary.total_datasets);
    println!("Total queries: {}", results.summary.total_queries);
    println!("\nBest methods per dataset:");
    for (dataset, method) in &results.summary.best_methods_per_dataset {
        println!("  {}: {}", dataset, method);
    }

    println!("\nAverage metrics across all datasets:");
    let mut method_avgs: Vec<_> = results.summary.method_averages.iter().collect();
    method_avgs.sort_by(|(_, a), (_, b)| {
        b.avg_ndcg_at_10
            .partial_cmp(&a.avg_ndcg_at_10)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    
    println!("{:<30} {:>10} {:>10} {:>10} {:>10}", "Method", "nDCG@10", "nDCG@100", "MAP", "MRR");
    println!("{}", "-".repeat(80));
    for (method_name, metrics) in method_avgs.iter().take(10) {
        println!(
            "{:<30} {:>10.4} {:>10.4} {:>10.4} {:>10.4}",
            method_name,
            metrics.avg_ndcg_at_10,
            metrics.avg_ndcg_at_100,
            metrics.avg_map,
            metrics.avg_mrr
        );
    }

    Ok(())
}

