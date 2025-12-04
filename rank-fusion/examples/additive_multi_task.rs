//! Example: Additive Multi-Task Fusion (ResFlow-style)
//!
//! Additive multi-task fusion combines scores from multiple tasks with configurable weights.
//! This is particularly useful for e-commerce ranking where you want to combine:
//! - Click-through rate (CTR)
//! - Click-to-conversion rate (CTCVR)
//! - Revenue per click
//!
//! ResFlow paper shows additive fusion outperforms multiplicative for multi-task ranking.

use rank_fusion::{
    additive_multi_task_with_config, AdditiveMultiTaskConfig, Normalization,
};

fn main() {
    // E-commerce example: Combining CTR and CTCVR
    // CTR: Click-through rate (high volume, low conversion)
    // CTCVR: Click-to-conversion rate (low volume, high value)

    // CTR scores (percentage of users who click)
    let ctr_scores = vec![
        ("item1", 0.05), // 5% CTR
        ("item2", 0.04),
        ("item3", 0.03),
        ("item4", 0.02),
    ];

    // CTCVR scores (percentage of clicks that convert)
    // These are typically much lower than CTR
    let ctcvr_scores = vec![
        ("item1", 0.02), // 2% conversion rate
        ("item2", 0.03), // item2 has better conversion!
        ("item3", 0.015),
        ("item4", 0.01),
    ];

    // Equal weights: both tasks matter equally
    let config_equal = AdditiveMultiTaskConfig::new((1.0, 1.0));
    let fused_equal = additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config_equal);

    println!("Equal weights (1.0, 1.0):");
    for (item, score) in &fused_equal {
        println!("  {}: {:.4}", item, score);
    }

    // ResFlow-style: Weight CTCVR 20× more (conversion is more valuable)
    // This is the recommended approach from the ResFlow paper
    let config_weighted =
        AdditiveMultiTaskConfig::new((1.0, 20.0)).with_normalization(Normalization::MinMax); // MinMax normalization

    let fused_weighted =
        additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config_weighted);

    println!("\nWeighted (1.0, 20.0) - ResFlow style:");
    println!(
        "  item1: {:.4} (CTR: 0.05, CTCVR: 0.02×20 = 0.40, total: 0.45)",
        fused_weighted[0].1
    );
    println!(
        "  item2: {:.4} (CTR: 0.04, CTCVR: 0.03×20 = 0.60, total: 0.64)",
        fused_weighted[1].1
    );
    // item2 wins because it has better conversion rate!

    // Different normalization methods
    println!("\nComparing normalization methods:");

    for norm in [
        Normalization::ZScore,
        Normalization::MinMax,
        Normalization::Sum,
    ] {
        let config = AdditiveMultiTaskConfig::new((1.0, 1.0)).with_normalization(norm);
        let fused = additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config);
        println!(
            "  {:?}: top item = {} (score: {:.4})",
            norm, fused[0].0, fused[0].1
        );
    }

    // Note: For 3+ lists, use additive_multi_task_multi with a slice
    println!("\nMulti-list fusion example:");
    println!("  Use additive_multi_task_multi() for 3+ lists");
}
