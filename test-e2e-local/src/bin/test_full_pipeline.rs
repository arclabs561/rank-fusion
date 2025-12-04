//! E2E test: Full pipeline - fusion + refine + eval.
//!
//! Simulates a complete RAG pipeline:
//! 1. Multiple retrievers (BM25, dense)
//! 2. Fuse results
//! 3. Refine with ColBERT
//! 4. Evaluate with rank-eval

use rank_fusion::{rrf, combsum};
use rank_refine::colbert;
use rank_eval::binary::ndcg_at_k;
use rank_eval::graded::compute_ndcg;
use std::collections::{HashMap, HashSet};

fn main() {
    println!("Testing full pipeline: fusion → refine → eval...");
    
    // Step 1: Simulate retrieval from multiple sources
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
    
    println!("✅ Retrieved {} BM25 results, {} dense results", 
             bm25_results.len(), dense_results.len());
    
    // Step 2: Fuse results
    let fused = rrf(&bm25_results, &dense_results);
    assert!(!fused.is_empty());
    println!("✅ Fused to {} results", fused.len());
    
    // Step 3: Prepare for refinement (simulate embeddings)
    // In real usage, these would come from embedding models
    let query_tokens = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
    ];
    
    // Create document token embeddings for top fused results
    let mut doc_embeddings = Vec::new();
    for (doc_id, _) in fused.iter().take(3) {
        let tokens = match doc_id.as_str() {
            "doc1" => vec![vec![0.9, 0.1, 0.0], vec![0.1, 0.9, 0.0]],
            "doc2" => vec![vec![0.95, 0.05, 0.0], vec![0.05, 0.95, 0.0]],
            "doc3" => vec![vec![0.8, 0.2, 0.0], vec![0.2, 0.8, 0.0]],
            _ => vec![vec![0.5, 0.5, 0.0], vec![0.5, 0.5, 0.0]],
        };
        doc_embeddings.push((doc_id.clone(), tokens));
    }
    
    // Refine with ColBERT
    let refined = colbert::rank(&query_tokens, &doc_embeddings);
    assert!(!refined.is_empty());
    println!("✅ Refined to {} results", refined.len());
    
    // Step 4: Evaluate
    let ranked: Vec<String> = refined.iter().map(|(id, _)| id.clone()).collect();
    let relevant: HashSet<String> = ["doc1", "doc2"].iter()
        .map(|s| s.to_string())
        .collect();
    
    let ndcg = ndcg_at_k(&ranked, &relevant, 10);
    assert!(ndcg >= 0.0 && ndcg <= 1.0);
    println!("✅ Binary nDCG@10: {:.4}", ndcg);
    
    // Graded evaluation
    let mut qrels = HashMap::new();
    qrels.insert("doc1".to_string(), 2);
    qrels.insert("doc2".to_string(), 1);
    
    let refined_graded: Vec<(String, f32)> = refined.iter()
        .map(|(id, score)| (id.clone(), *score))
        .collect();
    
    let graded_ndcg = compute_ndcg(&refined_graded, &qrels, 10);
    assert!(graded_ndcg >= 0.0 && graded_ndcg <= 1.0);
    println!("✅ Graded nDCG@10: {:.4}", graded_ndcg);
    
    // Alternative: Use CombSUM instead of RRF
    let fused_combsum = combsum(&bm25_results, &dense_results);
    let ranked_combsum: Vec<String> = fused_combsum.iter()
        .take(3)
        .map(|(id, _)| id.clone())
        .collect();
    
    let ndcg_combsum = ndcg_at_k(&ranked_combsum, &relevant, 10);
    println!("✅ CombSUM nDCG@10: {:.4}", ndcg_combsum);
    
    println!("\n✅ Full pipeline test passed!");
    println!("   Pipeline: Retrieve → Fuse → Refine → Evaluate");
}

