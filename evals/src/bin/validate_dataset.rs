//! Binary for validating dataset files.

use anyhow::{Context, Result};
use clap::Parser;
use rank_eval::dataset::{print_validation_report, validate_dataset};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "validate-dataset")]
#[command(about = "Validate TREC format dataset files")]
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

    println!("Validating dataset...");
    println!("  Runs: {:?}", args.runs);
    println!("  Qrels: {:?}", args.qrels);

    let result = validate_dataset(&args.runs, &args.qrels)
        .with_context(|| "Dataset validation failed")?;

    print_validation_report(&result);

    if let Some(json_path) = args.json {
        let json = serde_json::to_string_pretty(&result)?;
        std::fs::write(&json_path, json)
            .with_context(|| format!("Failed to write JSON report: {:?}", json_path))?;
        println!("\nJSON report written to: {:?}", json_path);
    }

    if !result.is_valid {
        std::process::exit(1);
    }

    Ok(())
}

