//! Example: Comparing score-based fusion methods.
//!
//! DBSF (z-score) vs CombSUM (min-max) when distributions differ.
//!
//! Run: `cargo run --example score_normalization`

use rank_fusion::{combsum, dbsf};

fn main() {
    // BM25: scores typically 0-30
    let bm25 = vec![
        ("doc_a", 28.5),
        ("doc_b", 25.0),
        ("doc_c", 12.0),
        ("doc_d", 8.0),
    ];

    // Dense: cosine similarity 0-1
    let dense = vec![
        ("doc_b", 0.98),
        ("doc_c", 0.92),
        ("doc_e", 0.85),
        ("doc_a", 0.78),
    ];

    println!("BM25 (scores 0-30): {:?}", bm25);
    println!("Dense (scores 0-1): {:?}\n", dense);

    // CombSUM: min-max normalization
    let combsum_result = combsum(&bm25, &dense);
    println!("CombSUM (min-max normalization):");
    for (id, score) in combsum_result.iter().take(5) {
        println!("  {id}: {score:.4}");
    }

    // DBSF: z-score normalization, better for different distributions
    let dbsf_result = dbsf(&bm25, &dense);
    println!("\nDBSF (z-score normalization):");
    for (id, score) in dbsf_result.iter().take(5) {
        println!("  {id}: {score:.4}");
    }

    println!("\nNote: DBSF uses z-score normalization (mean/std) which handles");
    println!("different score distributions better than min-max normalization.");
}

