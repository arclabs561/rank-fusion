//! Example: Standardized Fusion (ERANK-style)
//!
//! Standardized fusion uses z-score normalization to handle different score distributions.
//! This is particularly useful when combining results from different retrieval systems
//! that may have different score scales or distributions.

use rank_fusion::{standardized, standardized_with_config, StandardizedConfig};

fn main() {
    // Example: Combining BM25 and dense retrieval results
    // BM25 scores are typically in [0, 10+], while dense similarity is in [-1, 1]

    // BM25 results (higher scores = better)
    let bm25_results = vec![("doc1", 8.5), ("doc2", 7.2), ("doc3", 6.1), ("doc4", 5.8)];

    // Dense retrieval results (cosine similarity, typically [-1, 1])
    let dense_results = vec![
        ("doc1", 0.85),
        ("doc3", 0.92), // doc3 ranks higher in dense
        ("doc2", 0.78),
        ("doc5", 0.65), // doc5 only in dense
    ];

    // Default standardized fusion (clips z-scores to [-3.0, 3.0])
    let fused = standardized(&bm25_results, &dense_results);
    println!("Default standardized fusion:");
    for (doc, score) in &fused {
        println!("  {}: {:.4}", doc, score);
    }

    // Tighter clipping for more aggressive outlier handling
    let config = StandardizedConfig::new((-2.0, 2.0)) // Tighter range
        .with_top_k(3); // Only top 3 results

    let fused_tight = standardized_with_config(&bm25_results, &dense_results, config);
    println!("\nTight clipping ([-2, 2]) with top_k=3:");
    for (doc, score) in &fused_tight {
        println!("  {}: {:.4}", doc, score);
    }

    // Example: Handling negative scores (e.g., after centering)
    let negative_scores = vec![("doc1", -0.5), ("doc2", -0.7), ("doc3", -0.9)];

    let positive_scores = vec![("doc1", 0.9), ("doc2", 0.8), ("doc4", 0.7)];

    println!("\nHandling negative scores:");
    let fused_negative = standardized(&negative_scores, &positive_scores);
    for (doc, score) in &fused_negative {
        println!("  {}: {:.4}", doc, score);
    }
}
