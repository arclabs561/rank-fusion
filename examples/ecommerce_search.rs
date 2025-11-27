//! E-commerce multi-signal fusion.
//!
//! Combine multiple ranking signals for product search:
//! text relevance, popularity, recency, personalization.
//!
//! Run: `cargo run --example ecommerce_search`

use rank_fusion::{combsum_multi, rrf_multi, weighted_multi, FusionConfig, RrfConfig};

fn main() {
    println!("=== E-commerce Multi-Signal Fusion ===\n");

    let query = "wireless headphones";
    println!("Query: \"{query}\"\n");

    // Text relevance (BM25 or dense)
    let text_relevance: Vec<(u32, f32)> = vec![
        (1001, 0.95), // Sony WH-1000XM5
        (1002, 0.92), // AirPods Max
        (1003, 0.88), // Bose QC45
        (1004, 0.85), // Jabra Elite 85h
        (1005, 0.80), // Sennheiser Momentum 4
    ];

    // Popularity (sales rank, reviews)
    let popularity: Vec<(u32, f32)> = vec![
        (1002, 0.98), // AirPods Max (best seller)
        (1001, 0.90), // Sony
        (1006, 0.85), // Beats Studio3 (popular but less relevant)
        (1003, 0.75), // Bose
        (1007, 0.70), // Budget option
    ];

    // Recency (newer products boosted)
    let recency: Vec<(u32, f32)> = vec![
        (1005, 0.95), // Sennheiser (newest)
        (1001, 0.90), // Sony
        (1004, 0.85), // Jabra
        (1002, 0.70), // AirPods Max (older)
        (1003, 0.65), // Bose (oldest)
    ];

    // User personalization (based on history)
    let personalization: Vec<(u32, f32)> = vec![
        (1003, 0.90), // Bose (user bought Bose before)
        (1001, 0.80), // Sony
        (1005, 0.75), // Sennheiser
        (1002, 0.60), // AirPods
        (1004, 0.55), // Jabra
    ];

    println!("Individual signals:");
    println!(
        "  Text:    {:?}",
        text_relevance.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!(
        "  Popular: {:?}",
        popularity.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!(
        "  Recent:  {:?}",
        recency.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );
    println!(
        "  Personal:{:?}",
        personalization.iter().map(|(id, _)| id).collect::<Vec<_>>()
    );

    // RRF: Equal weight to all signals
    let signals: Vec<&[(u32, f32)]> = vec![
        &text_relevance,
        &popularity,
        &recency,
        &personalization,
    ];
    let rrf_result = rrf_multi(&signals, RrfConfig::default());

    println!("\nRRF fusion (equal weight):");
    for (id, score) in rrf_result.iter().take(5) {
        let name = product_name(*id);
        println!("  {name} (id={id}): {score:.4}");
    }

    // Weighted: Prioritize relevance, then personalization
    // 40% text, 25% personal, 20% popular, 15% recent
    let weighted_lists: Vec<(&[(u32, f32)], f32)> = vec![
        (&text_relevance[..], 0.40),
        (&personalization[..], 0.25),
        (&popularity[..], 0.20),
        (&recency[..], 0.15),
    ];
    let weighted_result = weighted_multi(&weighted_lists, true, None).unwrap();

    println!("\nWeighted (40% text, 25% personal, 20% pop, 15% recent):");
    for (id, score) in weighted_result.iter().take(5) {
        let name = product_name(*id);
        println!("  {name} (id={id}): {score:.4}");
    }

    // CombSUM: Simple score addition (after normalization)
    let combsum_result = combsum_multi(&signals, FusionConfig::default());

    println!("\nCombSUM (normalized sum):");
    for (id, score) in combsum_result.iter().take(5) {
        let name = product_name(*id);
        println!("  {name} (id={id}): {score:.4}");
    }

    // When to use which:
    // - RRF: Don't know relative signal quality, or scores are on different scales
    // - Weighted: You've measured signal quality (e.g., via A/B test or offline eval)
    // - CombSUM: Scores are already normalized to same scale (e.g., all [0,1])
}

fn product_name(id: u32) -> &'static str {
    match id {
        1001 => "Sony WH-1000XM5",
        1002 => "AirPods Max",
        1003 => "Bose QC45",
        1004 => "Jabra Elite 85h",
        1005 => "Sennheiser M4",
        1006 => "Beats Studio3",
        1007 => "Budget HP",
        _ => "Unknown",
    }
}
