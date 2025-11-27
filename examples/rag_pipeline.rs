//! RAG retrieval fusion pipeline.
//!
//! Combine dense vector search with BM25 keyword search
//! for better retrieval than either alone.
//!
//! Run: `cargo run --example rag_pipeline`

use rank_fusion::{rrf, rrf_with_config, weighted, RrfConfig, WeightedConfig};

fn main() {
    println!("=== RAG Retrieval Fusion ===\n");

    let query = "How does Rust prevent memory leaks?";
    println!("Query: \"{query}\"\n");

    // Dense vector search results (semantic similarity)
    // Returns (doc_id, similarity_score)
    let dense_results: Vec<(u32, f32)> = vec![
        (101, 0.92), // "Rust ownership model"
        (205, 0.89), // "Memory safety in Rust"
        (342, 0.85), // "Smart pointers in Rust"
        (156, 0.82), // "RAII pattern"
        (289, 0.78), // "Garbage collection alternatives"
    ];

    // BM25 keyword search results
    let bm25_results: Vec<(u32, f32)> = vec![
        (205, 12.5), // "Memory safety in Rust" (keyword match)
        (478, 11.2), // "Rust memory allocation"
        (101, 10.8), // "Rust ownership model"
        (512, 9.4),  // "Preventing leaks in C++"
        (342, 8.7),  // "Smart pointers in Rust"
    ];

    println!("Dense results (semantic):");
    for (id, score) in &dense_results {
        println!("  doc_{id}: {score:.2}");
    }

    println!("\nBM25 results (keyword):");
    for (id, score) in &bm25_results {
        println!("  doc_{id}: {score:.1}");
    }

    // RRF fusion (rank-based, ignores score magnitudes)
    let rrf_fused = rrf(&dense_results, &bm25_results);

    println!("\nRRF fusion (k=60):");
    for (id, score) in rrf_fused.iter().take(5) {
        println!("  doc_{id}: {score:.4}");
    }

    // RRF with custom k
    // k controls how much top positions dominate:
    //   k=10: position 0 gets 1/10=0.10, position 5 gets 1/15=0.067 (1.5x difference)
    //   k=60: position 0 gets 1/60=0.017, position 5 gets 1/65=0.015 (1.1x difference)
    // Lower k = sharper preference for top ranks
    let rrf_topk = rrf_with_config(&dense_results, &bm25_results, RrfConfig::new(10));

    println!("\nRRF (k=10, top-heavy):");
    for (id, score) in rrf_topk.iter().take(5) {
        println!("  doc_{id}: {score:.4}");
    }

    // Weighted fusion (when you trust one retriever more)
    // 70% dense, 30% BM25
    let weighted_fused = weighted(&dense_results, &bm25_results, WeightedConfig::new(0.7, 0.3));

    println!("\nWeighted (70% dense, 30% BM25):");
    for (id, score) in weighted_fused.iter().take(5) {
        println!("  doc_{id}: {score:.4}");
    }

    // Observations:
    // - doc_205 and doc_101 appear in both lists, so RRF boosted them
    // - RRF ignores that BM25 scores (12.5) are much larger than dense (0.92)
    // - Weighted fusion is useful when you've measured retriever quality
}
