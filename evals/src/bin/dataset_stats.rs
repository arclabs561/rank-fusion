//! Binary for computing and displaying dataset statistics.

use anyhow::{Context, Result};
use clap::Parser;
use rank_eval::dataset::{compute_comprehensive_stats, print_statistics_report};
use rank_eval::trec::{load_qrels, load_trec_runs};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "dataset-stats")]
#[command(about = "Compute comprehensive statistics for a dataset")]
struct Args {
    /// Path to TREC runs file
    #[arg(long, required = true)]
    runs: PathBuf,

    /// Path to TREC qrels file
    #[arg(long, required = true)]
    qrels: PathBuf,

    /// Output JSON report
    #[arg(long)]
    json: Option<PathBuf>,
}

fn main() -> Result<()> {
    let args = Args::parse();

    println!("Loading dataset...");
    println!("  Runs: {:?}", args.runs);
    println!("  Qrels: {:?}", args.qrels);

    let runs = load_trec_runs(&args.runs)
        .with_context(|| format!("Failed to load runs: {:?}", args.runs))?;
    
    let qrels = load_qrels(&args.qrels)
        .with_context(|| format!("Failed to load qrels: {:?}", args.qrels))?;

    println!("Computing statistics...");
    let stats = compute_comprehensive_stats(&runs, &qrels);

    print_statistics_report(&stats);

    if let Some(json_path) = args.json {
        let json = serde_json::to_string_pretty(&stats)?;
        std::fs::write(&json_path, json)
            .with_context(|| format!("Failed to write JSON report: {:?}", json_path))?;
        println!("JSON report written to: {:?}", json_path);
    }

    Ok(())
}

