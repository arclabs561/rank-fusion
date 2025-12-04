//! Dataset format converters.
//!
//! Converts datasets from various formats (HuggingFace, JSON, etc.) to TREC format
//! for use with the evaluation system.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

/// Convert HuggingFace dataset format to TREC runs format.
///
/// Assumes dataset has structure with query_id, doc_id, score fields.
/// Groups by query_id and ranks documents within each query by score (descending).
pub fn convert_hf_to_trec_runs(
    data: &[HuggingFaceExample],
    output_path: impl AsRef<Path>,
    run_tag: &str,
) -> Result<()> {
    // Group by query_id
    let mut by_query: std::collections::HashMap<String, Vec<&HuggingFaceExample>> = 
        std::collections::HashMap::new();
    
    for example in data {
        by_query
            .entry(example.query_id.clone())
            .or_insert_with(Vec::new)
            .push(example);
    }

    // Sort each query's documents by score (descending)
    for examples in by_query.values_mut() {
        examples.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    }

    let file = File::create(output_path.as_ref())
        .with_context(|| format!("Failed to create output file: {:?}", output_path.as_ref()))?;
    let mut writer = BufWriter::new(file);

    // Sort queries for consistent output
    let mut query_ids: Vec<String> = by_query.keys().cloned().collect();
    query_ids.sort();

    for query_id in query_ids {
        let examples = &by_query[&query_id];
        for (rank, example) in examples.iter().enumerate() {
            writeln!(
                writer,
                "{} Q0 {} {} {:.6} {}",
                query_id,
                example.doc_id,
                rank + 1,
                example.score,
                run_tag
            )?;
        }
    }

    writer.flush()?;
    Ok(())
}

/// Convert HuggingFace dataset format to TREC qrels format.
pub fn convert_hf_to_trec_qrels(
    data: &[HuggingFaceQrel],
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let file = File::create(output_path.as_ref())
        .with_context(|| format!("Failed to create output file: {:?}", output_path.as_ref()))?;
    let mut writer = BufWriter::new(file);

    // Sort for consistent output
    let mut sorted_data: Vec<&HuggingFaceQrel> = data.iter().collect();
    sorted_data.sort_by(|a, b| {
        a.query_id
            .cmp(&b.query_id)
            .then_with(|| a.doc_id.cmp(&b.doc_id))
    });

    for qrel in sorted_data {
        writeln!(
            writer,
            "{} 0 {} {}",
            qrel.query_id, qrel.doc_id, qrel.relevance
        )?;
    }

    writer.flush()?;
    Ok(())
}

/// HuggingFace dataset example structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceExample {
    pub query_id: String,
    pub doc_id: String,
    pub score: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub rank: Option<usize>,
}

/// HuggingFace qrel structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HuggingFaceQrel {
    pub query_id: String,
    pub doc_id: String,
    pub relevance: u32,
}

/// Convert JSON lines format to TREC runs.
pub fn convert_jsonl_to_trec_runs(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
    run_tag: &str,
) -> Result<()> {
    let content = std::fs::read_to_string(input_path.as_ref())
        .with_context(|| format!("Failed to read input: {:?}", input_path.as_ref()))?;

    let mut examples = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let example: HuggingFaceExample = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse JSON line {}: {}", line_num + 1, line))?;
        examples.push(example);
    }

    if examples.is_empty() {
        anyhow::bail!("No valid examples found in input file");
    }

    convert_hf_to_trec_runs(&examples, output_path, run_tag)
}

/// Convert JSON lines format to TREC qrels.
pub fn convert_jsonl_to_trec_qrels(
    input_path: impl AsRef<Path>,
    output_path: impl AsRef<Path>,
) -> Result<()> {
    let content = std::fs::read_to_string(input_path.as_ref())
        .with_context(|| format!("Failed to read input: {:?}", input_path.as_ref()))?;

    let mut qrels = Vec::new();
    for (line_num, line) in content.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        let qrel: HuggingFaceQrel = serde_json::from_str(line)
            .with_context(|| format!("Failed to parse JSON line {}: {}", line_num + 1, line))?;
        qrels.push(qrel);
    }

    if qrels.is_empty() {
        anyhow::bail!("No valid qrels found in input file");
    }

    convert_hf_to_trec_qrels(&qrels, output_path)
}

/// Convert BEIR format qrels to TREC format.
///
/// BEIR datasets typically come in a specific format that needs conversion.
/// Note: This only converts qrels. Runs must be generated separately by running
/// a retrieval system (BM25, dense retrieval, etc.) on the BEIR corpus.
pub fn convert_beir_qrels_to_trec(
    qrels_path: impl AsRef<Path>,
    qrels_output: impl AsRef<Path>,
) -> Result<()> {
    let qrels_content = std::fs::read_to_string(qrels_path.as_ref())
        .with_context(|| format!("Failed to read BEIR qrels: {:?}", qrels_path.as_ref()))?;
    
    let beir_qrels: HashMap<String, HashMap<String, u32>> = serde_json::from_str(&qrels_content)
        .context("Failed to parse BEIR qrels JSON")?;

    // Convert qrels
    let mut qrels_file = File::create(qrels_output.as_ref())
        .with_context(|| format!("Failed to create output qrels: {:?}", qrels_output.as_ref()))?;
    
    // Sort for consistent output
    let mut query_ids: Vec<String> = beir_qrels.keys().cloned().collect();
    query_ids.sort();
    
    for query_id in query_ids {
        let doc_relevance = &beir_qrels[&query_id];
        let mut doc_ids: Vec<String> = doc_relevance.keys().cloned().collect();
        doc_ids.sort();
        
        for doc_id in doc_ids {
            let relevance = doc_relevance[&doc_id];
            writeln!(qrels_file, "{} 0 {} {}", query_id, doc_id, relevance)
                .with_context(|| format!("Failed to write qrel line"))?;
        }
    }

    Ok(())
}

/// BEIR document structure.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BeirDocument {
    pub title: Option<String>,
    pub text: String,
}

/// Dataset conversion configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionConfig {
    pub input_format: String, // "huggingface", "beir", "jsonl", "trec"
    pub output_format: String, // "trec"
    pub input_path: PathBuf,
    pub output_path: PathBuf,
    pub run_tag: Option<String>,
}

/// Convert dataset based on configuration.
pub fn convert_dataset(config: &ConversionConfig) -> Result<()> {
    match config.input_format.as_str() {
        "jsonl" => {
            if config.output_path.to_string_lossy().contains("qrels") {
                convert_jsonl_to_trec_qrels(&config.input_path, &config.output_path)?;
            } else {
                let run_tag = config.run_tag.as_deref().unwrap_or("converted");
                convert_jsonl_to_trec_runs(&config.input_path, &config.output_path, run_tag)?;
            }
        }
        "trec" => {
            // Already in TREC format, just copy
            std::fs::copy(&config.input_path, &config.output_path)
                .with_context(|| format!("Failed to copy TREC file: {:?}", config.input_path))?;
        }
        _ => {
            anyhow::bail!("Unsupported input format: {}", config.input_format);
        }
    }
    Ok(())
}

/// Validate TREC runs file format.
pub fn validate_trec_runs(path: impl AsRef<Path>) -> Result<ValidationResult> {
    use crate::real_world::load_trec_runs;
    
    let runs = load_trec_runs(path.as_ref())?;
    
    if runs.is_empty() {
        return Ok(ValidationResult {
            is_valid: false,
            errors: vec!["File is empty".to_string()],
            warnings: vec![],
            stats: None,
        });
    }

    let errors = Vec::new();
    let mut warnings = Vec::new();
    
    // Check for duplicate query-doc pairs
    let mut seen = std::collections::HashSet::new();
    for run in &runs {
        let key = format!("{}:{}:{}", run.query_id, run.doc_id, run.run_tag);
        if !seen.insert(key.clone()) {
            warnings.push(format!("Duplicate entry: {}", key));
        }
    }

    // Check rank ordering within queries
    let mut by_query: HashMap<String, Vec<&crate::real_world::TrecRun>> = HashMap::new();
    for run in &runs {
        by_query.entry(run.query_id.clone()).or_insert_with(Vec::new).push(run);
    }

    for (query_id, query_runs) in &by_query {
        let mut sorted = query_runs.clone();
        sorted.sort_by_key(|r| r.rank);
        
        for (expected_rank, run) in sorted.iter().enumerate() {
            if run.rank != expected_rank + 1 {
                warnings.push(format!(
                    "Query {}: rank {} not sequential (expected {})",
                    query_id, run.rank, expected_rank + 1
                ));
            }
        }
    }

    let stats = Some(ConversionStats {
        total_entries: runs.len(),
        unique_queries: by_query.len(),
        unique_documents: runs.iter().map(|r| &r.doc_id).collect::<std::collections::HashSet<_>>().len(),
        unique_run_tags: runs.iter().map(|r| &r.run_tag).collect::<std::collections::HashSet<_>>().len(),
    });

    Ok(ValidationResult {
        is_valid: errors.is_empty(),
        errors,
        warnings,
        stats,
    })
}

/// Validation result.
#[derive(Debug, Clone)]
pub struct ValidationResult {
    pub is_valid: bool,
    pub errors: Vec<String>,
    pub warnings: Vec<String>,
    pub stats: Option<ConversionStats>,
}

/// Conversion statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversionStats {
    pub total_entries: usize,
    pub unique_queries: usize,
    pub unique_documents: usize,
    pub unique_run_tags: usize,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_convert_hf_to_trec_runs() {
        let examples = vec![
            HuggingFaceExample {
                query_id: "1".to_string(),
                doc_id: "doc1".to_string(),
                score: 0.95,
                rank: None,
            },
            HuggingFaceExample {
                query_id: "1".to_string(),
                doc_id: "doc2".to_string(),
                score: 0.87,
                rank: None,
            },
            HuggingFaceExample {
                query_id: "2".to_string(),
                doc_id: "doc3".to_string(),
                score: 0.92,
                rank: None,
            },
        ];

        let temp_file = std::env::temp_dir().join("test_runs.txt");
        convert_hf_to_trec_runs(&examples, &temp_file, "test_run").unwrap();

        let content = std::fs::read_to_string(&temp_file).unwrap();
        assert!(content.contains("1 Q0 doc1 1"));
        assert!(content.contains("1 Q0 doc2 2")); // Should be ranked 2nd for query 1
        assert!(content.contains("2 Q0 doc3 1")); // Should be ranked 1st for query 2
        assert!(content.contains("test_run"));

        std::fs::remove_file(&temp_file).ok();
    }

    #[test]
    fn test_convert_hf_to_trec_runs_groups_by_query() {
        // Test that documents are properly grouped by query and ranked within each query
        let examples = vec![
            HuggingFaceExample {
                query_id: "q1".to_string(),
                doc_id: "d1".to_string(),
                score: 0.5,
                rank: None,
            },
            HuggingFaceExample {
                query_id: "q2".to_string(),
                doc_id: "d2".to_string(),
                score: 0.9,
                rank: None,
            },
            HuggingFaceExample {
                query_id: "q1".to_string(),
                doc_id: "d3".to_string(),
                score: 0.8,
                rank: None,
            },
        ];

        let temp_file = std::env::temp_dir().join("test_runs_grouped.txt");
        convert_hf_to_trec_runs(&examples, &temp_file, "test").unwrap();

        let content = std::fs::read_to_string(&temp_file).unwrap();
        let lines: Vec<&str> = content.lines().collect();
        
        // Query q1 should have d3 ranked 1 (score 0.8) and d1 ranked 2 (score 0.5)
        assert!(lines.iter().any(|l| l.contains("q1 Q0 d3 1")));
        assert!(lines.iter().any(|l| l.contains("q1 Q0 d1 2")));
        // Query q2 should have d2 ranked 1
        assert!(lines.iter().any(|l| l.contains("q2 Q0 d2 1")));

        std::fs::remove_file(&temp_file).ok();
    }
}
