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

    // RRF with custom k (lower k = top ranks matter more)
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

    println!("\n=== Key Observations ===");
    println!("* doc_205 and doc_101 appear in both -> boosted by RRF");
    println!("* RRF ignores score magnitudes -> BM25's high scores don't dominate");
    println!("* Weighted fusion when you know retriever quality differs");
}
