//! E2E test: rank-fusion + rank-eval integration.
//!
//! Simulates a real-world scenario where a user fuses results and evaluates them.

use rank_fusion::{rrf, combsum};
use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k, mrr};
use rank_eval::graded::{compute_ndcg, compute_map};
use std::collections::{HashMap, HashSet};

fn main() {
    println!("Testing rank-fusion + rank-eval integration...");
    
    // Simulate retrieval results
    let bm25_results = vec![
        ("doc1".to_string(), 12.5),
        ("doc2".to_string(), 11.0),
        ("doc3".to_string(), 10.5),
        ("doc4".to_string(), 9.0),
    ];
    
    let dense_results = vec![
        ("doc2".to_string(), 0.95),
        ("doc1".to_string(), 0.88),
        ("doc5".to_string(), 0.82),
        ("doc3".to_string(), 0.75),
    ];
    
    // Fuse results
    let fused = rrf(&bm25_results, &dense_results);
    assert!(!fused.is_empty());
    println!("✅ Fused {} results", fused.len());
    
    // Extract ranked list for evaluation
    let ranked: Vec<String> = fused.iter().map(|(id, _)| id.clone()).collect();
    
    // Binary relevance
    let relevant: HashSet<String> = ["doc1", "doc2", "doc3"].iter()
        .map(|s| s.to_string())
        .collect();
    
    // Compute binary metrics
    let ndcg_10 = ndcg_at_k(&ranked, &relevant, 10);
    let p_10 = precision_at_k(&ranked, &relevant, 10);
    let r_10 = recall_at_k(&ranked, &relevant, 10);
    let mrr_score = mrr(&ranked, &relevant);
    
    assert!(ndcg_10 >= 0.0 && ndcg_10 <= 1.0);
    assert!(p_10 >= 0.0 && p_10 <= 1.0);
    assert!(r_10 >= 0.0 && r_10 <= 1.0);
    assert!(mrr_score >= 0.0 && mrr_score <= 1.0);
    
    println!("✅ Binary metrics:");
    println!("   nDCG@10: {:.4}", ndcg_10);
    println!("   P@10: {:.4}", p_10);
    println!("   R@10: {:.4}", r_10);
    println!("   MRR: {:.4}", mrr_score);
    
    // Graded relevance
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2); // Highly relevant
    qrels.insert("doc2".to_string(), 1); // Relevant
    qrels.insert("doc3".to_string(), 0); // Not relevant
    
    let graded_ndcg = compute_ndcg(&fused, &qrels, 10);
    let map_score = compute_map(&fused, &qrels);
    
    assert!(graded_ndcg >= 0.0 && graded_ndcg <= 1.0);
    assert!(map_score >= 0.0 && map_score <= 1.0);
    
    println!("✅ Graded metrics:");
    println!("   nDCG@10: {:.4}", graded_ndcg);
    println!("   MAP: {:.4}", map_score);
    
    // Test with CombSUM
    let fused_combsum = combsum(&bm25_results, &dense_results);
    let ranked_combsum: Vec<String> = fused_combsum.iter().map(|(id, _)| id.clone()).collect();
    let ndcg_combsum = ndcg_at_k(&ranked_combsum, &relevant, 10);
    
    assert!(ndcg_combsum >= 0.0 && ndcg_combsum <= 1.0);
    println!("✅ CombSUM nDCG@10: {:.4}", ndcg_combsum);
    
    println!("\n✅ All fusion-eval integration tests passed!");
}

