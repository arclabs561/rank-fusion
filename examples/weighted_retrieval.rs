//! Example: Weighted fusion with per-retriever trust.
//!
//! Run: `cargo run --example weighted_retrieval`

use rank_fusion::{rrf_weighted, weighted_multi, RrfConfig};

fn main() {
    // Scenario: E-commerce product search
    // - BM25 is good for exact product names
    // - Dense is good for semantic matching
    // - We trust BM25 more for this domain

    let bm25 = vec![
        ("iphone_15_pro", 28.0),
        ("iphone_15", 22.0),
        ("samsung_s24", 15.0),
    ];

    let dense = vec![
        ("samsung_s24", 0.95),
        ("pixel_8", 0.91),
        ("iphone_15_pro", 0.88),
    ];

    println!("BM25: {:?}", bm25);
    println!("Dense: {:?}\n", dense);

    // Method 1: Weighted RRF (rank-based, with weights)
    let weights = [0.7, 0.3]; // 70% BM25, 30% dense
    let weighted_rrf = rrf_weighted(&[&bm25[..], &dense[..]], &weights, RrfConfig::default())
        .expect("valid weights");

    println!("Weighted RRF (70% BM25, 30% dense):");
    for (id, score) in &weighted_rrf {
        println!("  {id}: {score:.4}");
    }

    // Method 2: Weighted score fusion
    let weighted_score = weighted_multi(&[(&bm25, 0.7), (&dense, 0.3)], true, None)
        .expect("valid weights");

    println!("\nWeighted score fusion (70% BM25, 30% dense):");
    for (id, score) in &weighted_score {
        println!("  {id}: {score:.4}");
    }

    // Method 3: Equal weights (baseline)
    let equal_weighted = weighted_multi(&[(&bm25, 0.5), (&dense, 0.5)], true, None)
        .expect("valid weights");

    println!("\nEqual weights (50/50):");
    for (id, score) in &equal_weighted {
        println!("  {id}: {score:.4}");
    }

    println!("\nNote: BM25-heavy weighting favors 'iphone_15_pro' (exact match).");
    println!("Equal weights give more influence to dense retriever's semantic matches.");
}

