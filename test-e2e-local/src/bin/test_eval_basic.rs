//! E2E test: Basic rank-eval usage as a published crate.

use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k, mrr, average_precision};
use rank_eval::graded::{compute_ndcg, compute_map};
use rank_eval::trec::{load_trec_runs, load_qrels};
use std::collections::{HashMap, HashSet};
use std::io::Write;
use tempfile::TempDir;

fn main() {
    println!("Testing rank-eval as published crate...");
    
    // Test binary metrics
    let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
    let relevant: HashSet<&str> = ["doc1", "doc3", "doc5"].iter().cloned().collect();
    
    let ndcg = ndcg_at_k(&ranked, &relevant, 5);
    let precision = precision_at_k(&ranked, &relevant, 5);
    let recall = recall_at_k(&ranked, &relevant, 5);
    let mrr_score = mrr(&ranked, &relevant);
    let ap = average_precision(&ranked, &relevant);
    
    assert!(ndcg >= 0.0 && ndcg <= 1.0);
    assert!(precision >= 0.0 && precision <= 1.0);
    assert!(recall >= 0.0 && recall <= 1.0);
    assert!(mrr_score >= 0.0 && mrr_score <= 1.0);
    assert!(ap >= 0.0 && ap <= 1.0);
    
    println!("✅ Binary metrics:");
    println!("   nDCG@5: {:.4}", ndcg);
    println!("   P@5: {:.4}", precision);
    println!("   R@5: {:.4}", recall);
    println!("   MRR: {:.4}", mrr_score);
    println!("   AP: {:.4}", ap);
    
    // Test graded metrics
    let ranked_graded = vec![
        ("doc1".to_string(), 0.95),
        ("doc2".to_string(), 0.85),
        ("doc3".to_string(), 0.75),
    ];
    
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2);
    qrels.insert("doc2".to_string(), 1);
    qrels.insert("doc3".to_string(), 0);
    
    let graded_ndcg = compute_ndcg(&ranked_graded, &qrels, 10);
    let map_score = compute_map(&ranked_graded, &qrels);
    
    assert!(graded_ndcg >= 0.0 && graded_ndcg <= 1.0);
    assert!(map_score >= 0.0 && map_score <= 1.0);
    
    println!("✅ Graded metrics:");
    println!("   nDCG@10: {:.4}", graded_ndcg);
    println!("   MAP: {:.4}", map_score);
    
    // Test TREC parsing
    let temp_dir = TempDir::new().unwrap();
    let runs_file = temp_dir.path().join("runs.txt");
    let qrels_file = temp_dir.path().join("qrels.txt");
    
    // Write test TREC files
    let mut runs_writer = std::fs::File::create(&runs_file).unwrap();
    writeln!(runs_writer, "1 Q0 doc1 1 0.95 run1").unwrap();
    writeln!(runs_writer, "1 Q0 doc2 2 0.87 run1").unwrap();
    writeln!(runs_writer, "2 Q0 doc3 1 0.92 run1").unwrap();
    
    let mut qrels_writer = std::fs::File::create(&qrels_file).unwrap();
    writeln!(qrels_writer, "1 0 doc1 2").unwrap();
    writeln!(qrels_writer, "1 0 doc2 1").unwrap();
    writeln!(qrels_writer, "2 0 doc3 2").unwrap();
    
    // Parse TREC files
    let runs = load_trec_runs(&runs_file).unwrap();
    let qrels = load_qrels(&qrels_file).unwrap();
    
    assert_eq!(runs.len(), 3);
    assert_eq!(qrels.len(), 3);
    assert_eq!(runs[0].query_id, "1");
    assert_eq!(runs[0].doc_id, "doc1");
    assert_eq!(qrels[0].query_id, "1");
    assert_eq!(qrels[0].doc_id, "doc1");
    assert_eq!(qrels[0].relevance, 2);
    
    println!("✅ TREC parsing:");
    println!("   Loaded {} runs", runs.len());
    println!("   Loaded {} qrels", qrels.len());
    
    println!("\n✅ All rank-eval basic tests passed!");
}

