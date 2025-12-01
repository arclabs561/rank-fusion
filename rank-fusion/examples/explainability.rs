//! Example: Using explainability to debug rank fusion results.
//!
//! This example shows how to use the explainability API to understand
//! which retrievers contributed each document and why certain documents
//! ranked where they did.

use rank_fusion::explain::{analyze_consensus, attribute_top_k, rrf_explain, RetrieverId};
use rank_fusion::RrfConfig;

fn main() {
    // Simulate results from different retrievers
    let bm25_results = [
        ("doc_123", 87.5),
        ("doc_456", 82.3),
        ("doc_789", 78.1),
        ("doc_101", 75.0),
    ];

    let dense_results = [
        ("doc_456", 0.92),
        ("doc_123", 0.88),
        ("doc_999", 0.85),
        ("doc_101", 0.80),
    ];

    let sparse_results = [("doc_123", 0.95), ("doc_456", 0.90), ("doc_888", 0.85)];

    // Label each retriever
    let retrievers = [
        RetrieverId::new("bm25"),
        RetrieverId::new("dense"),
        RetrieverId::new("sparse"),
    ];

    // Fuse with explainability
    let explained = rrf_explain(
        &[&bm25_results[..], &dense_results[..], &sparse_results[..]],
        &retrievers,
        RrfConfig::default(),
    );

    println!("Top 5 fused results with explanations:\n");
    for (i, result) in explained.iter().take(5).enumerate() {
        println!(
            "{}. {} (score: {:.6}, rank: {})",
            i + 1,
            result.id,
            result.score,
            result.rank
        );
        println!(
            "   Consensus: {:.1}% of retrievers",
            result.explanation.consensus_score * 100.0
        );
        println!("   Sources:");
        for source in &result.explanation.sources {
            println!(
                "     - {}: rank {}, contribution {:.6}",
                source.retriever_id,
                source.original_rank.unwrap_or(999),
                source.contribution
            );
        }
        println!();
    }

    // Analyze consensus patterns
    println!("Consensus Analysis:\n");
    let consensus = analyze_consensus(&explained);

    println!("High consensus documents (in all retrievers):");
    for doc_id in &consensus.high_consensus {
        println!("  - {}", doc_id);
    }

    println!("\nSingle-source documents (only in one retriever):");
    for doc_id in &consensus.single_source {
        println!("  - {}", doc_id);
    }

    println!("\nRank disagreements (large spread across retrievers):");
    for (doc_id, rank_info) in &consensus.rank_disagreement {
        println!("  - {}: {:?}", doc_id, rank_info);
    }

    // Attribute top-k to retrievers
    println!("\nRetriever Attribution (top 5):\n");
    let attribution = attribute_top_k(&explained, 5);

    for (retriever_id, stats) in &attribution {
        println!("{}:", retriever_id);
        println!("  - Top-5 count: {}", stats.top_k_count);
        println!("  - Avg contribution: {:.6}", stats.avg_contribution);
        println!("  - Unique documents: {}", stats.unique_docs);
        println!();
    }
}
