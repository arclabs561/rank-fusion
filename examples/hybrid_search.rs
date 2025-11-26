//! Example: Hybrid search with RRF fusion.
//!
//! Run: `cargo run --example hybrid_search`

use rank_fusion::prelude::*;

fn main() {
    // Simulated BM25 results (lexical search)
    let bm25_results = vec![
        ("rust_book", 15.2),
        ("python_book", 12.1),
        ("go_book", 8.5),
        ("java_book", 5.0),
    ];

    // Simulated dense vector results (semantic search)
    let dense_results = vec![
        ("rust_book", 0.95),
        ("cpp_book", 0.88),
        ("python_book", 0.82),
        ("rust_tutorial", 0.78),
    ];

    println!("BM25 results: {:?}", bm25_results);
    println!("Dense results: {:?}\n", dense_results);

    // Method 1: RRF (best for incompatible score scales)
    let rrf_fused = rrf(
        bm25_results.clone(),
        dense_results.clone(),
        RrfConfig::default().with_top_k(5),
    );
    println!("RRF fusion (top 5):");
    for (id, score) in &rrf_fused {
        println!("  {id}: {score:.4}");
    }

    // Method 2: CombMNZ (rewards overlap)
    let combmnz_fused = combmnz(&bm25_results, &dense_results);
    println!("\nCombMNZ fusion:");
    for (id, score) in combmnz_fused.iter().take(5) {
        println!("  {id}: {score:.4}");
    }

    // Method 3: Weighted (trust dense more)
    let weighted_fused = weighted(
        &bm25_results,
        &dense_results,
        WeightedConfig::new(0.3, 0.7), // 30% BM25, 70% dense
    );
    println!("\nWeighted fusion (30% BM25, 70% dense):");
    for (id, score) in weighted_fused.iter().take(5) {
        println!("  {id}: {score:.4}");
    }

    // Method 4: Using FusionMethod enum
    use rank_fusion::FusionMethod;
    let method = FusionMethod::Rrf { k: 60 };
    let unified_fused = method.fuse(&bm25_results, &dense_results);
    println!("\nFusionMethod::Rrf:");
    for (id, score) in unified_fused.iter().take(3) {
        println!("  {id}: {score:.4}");
    }
}

