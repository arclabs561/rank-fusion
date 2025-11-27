//! Example: Fusing results from 3+ retrievers.
//!
//! Run: `cargo run --example multi_retriever`

use rank_fusion::{borda_multi, combmnz_multi, rrf_multi, FusionConfig, RrfConfig};

fn main() {
    // BM25 lexical search
    let bm25 = vec![
        ("doc_rust", 15.2),
        ("doc_python", 12.1),
        ("doc_go", 8.5),
    ];

    // Dense vector search (e.g., OpenAI embeddings)
    let dense = vec![
        ("doc_rust", 0.95),
        ("doc_cpp", 0.88),
        ("doc_python", 0.82),
    ];

    // Sparse vector search (e.g., SPLADE)
    let sparse = vec![
        ("doc_rust", 0.91),
        ("doc_python", 0.85),
        ("doc_java", 0.72),
    ];

    let lists = vec![&bm25[..], &dense[..], &sparse[..]];

    println!("Input:");
    println!("  BM25:   {:?}", bm25);
    println!("  Dense:  {:?}", dense);
    println!("  Sparse: {:?}\n", sparse);

    // RRF: Rank-based, ignores scores
    let rrf_result = rrf_multi(&lists, RrfConfig::default());
    println!("RRF (k=60):");
    for (id, score) in &rrf_result {
        println!("  {id}: {score:.4}");
    }

    // CombMNZ: Rewards items appearing in multiple lists
    let combmnz_result = combmnz_multi(&lists, FusionConfig::default());
    println!("\nCombMNZ:");
    for (id, score) in &combmnz_result {
        println!("  {id}: {score:.4}");
    }

    // Borda: Position-based voting
    let borda_result = borda_multi(&lists, FusionConfig::default());
    println!("\nBorda:");
    for (id, score) in &borda_result {
        println!("  {id}: {score:.4}");
    }

    // Note: doc_rust appears in all 3 lists and tops all fusion methods
    println!("\nObservation: 'doc_rust' appears in all 3 lists and wins in all methods.");
}

