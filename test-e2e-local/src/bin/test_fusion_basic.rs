//! E2E test: Basic rank-fusion usage as a published crate.
//!
//! This simulates how a user would use rank-fusion after installing it from crates.io.

use rank_fusion::{rrf, isr, combsum, combmnz, borda, dbsf, weighted, standardized_with_config, additive_multi_task_with_config};
use rank_fusion::{WeightedConfig, StandardizedConfig, AdditiveMultiTaskConfig};

fn main() {
    println!("Testing rank-fusion as published crate...");
    
    // Simulate two retrieval results
    let bm25_results = vec![
        ("doc1".to_string(), 12.5),
        ("doc2".to_string(), 11.0),
        ("doc3".to_string(), 10.5),
    ];
    
    let dense_results = vec![
        ("doc2".to_string(), 0.95),
        ("doc1".to_string(), 0.88),
        ("doc4".to_string(), 0.82),
    ];
    
    // Test RRF
    let rrf_fused = rrf(&bm25_results, &dense_results);
    assert!(!rrf_fused.is_empty(), "RRF should produce results");
    assert!(rrf_fused[0].0 == "doc2" || rrf_fused[0].0 == "doc1", "Top result should be doc1 or doc2");
    println!("✅ RRF: {} results", rrf_fused.len());
    
    // Test ISR
    let isr_fused = isr(&bm25_results, &dense_results);
    assert!(!isr_fused.is_empty());
    println!("✅ ISR: {} results", isr_fused.len());
    
    // Test CombSUM
    let combsum_fused = combsum(&bm25_results, &dense_results);
    assert!(!combsum_fused.is_empty());
    println!("✅ CombSUM: {} results", combsum_fused.len());
    
    // Test CombMNZ
    let combmnz_fused = combmnz(&bm25_results, &dense_results);
    assert!(!combmnz_fused.is_empty());
    println!("✅ CombMNZ: {} results", combmnz_fused.len());
    
    // Test Borda
    let borda_fused = borda(&bm25_results, &dense_results);
    assert!(!borda_fused.is_empty());
    println!("✅ Borda: {} results", borda_fused.len());
    
    // Test DBSF
    let dbsf_fused = dbsf(&bm25_results, &dense_results);
    assert!(!dbsf_fused.is_empty());
    println!("✅ DBSF: {} results", dbsf_fused.len());
    
    // Test Weighted
    let weighted_fused = weighted(
        &bm25_results,
        &dense_results,
        WeightedConfig::new(0.7, 0.3)
    );
    assert!(!weighted_fused.is_empty());
    println!("✅ Weighted: {} results", weighted_fused.len());
    
    // Test Standardized
    let standardized_fused = standardized_with_config(
        &bm25_results,
        &dense_results,
        StandardizedConfig::default()
    );
    assert!(!standardized_fused.is_empty());
    println!("✅ Standardized: {} results", standardized_fused.len());
    
    // Test Additive Multi-Task
    let additive_fused = additive_multi_task_with_config(
        &bm25_results,
        &dense_results,
        AdditiveMultiTaskConfig::new((1.0, 1.0))
    );
    assert!(!additive_fused.is_empty());
    println!("✅ Additive Multi-Task: {} results", additive_fused.len());
    
    // Test multi-run fusion
    let run3 = vec![("doc1".to_string(), 0.92), ("doc3".to_string(), 0.85)];
    let multi_fused = rank_fusion::rrf_multi(&[&bm25_results, &dense_results, &run3], rank_fusion::RrfConfig::default());
    assert!(!multi_fused.is_empty());
    println!("✅ Multi-run fusion: {} results", multi_fused.len());
    
    println!("\n✅ All rank-fusion basic tests passed!");
}

