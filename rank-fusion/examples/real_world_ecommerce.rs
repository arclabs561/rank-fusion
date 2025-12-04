//! Real-world integration example: E-commerce product ranking.
//!
//! This example demonstrates how to use `additive_multi_task` fusion for
//! e-commerce ranking, combining CTR (click-through rate) and CTCVR
//! (click-through conversion rate) signals.
//!
//! Based on ResFlow paper (arXiv:2411.09705) showing additive fusion
//! outperforms multiplicative for e-commerce.
//!
//! Run: `cargo run --example real_world_ecommerce`

use rank_fusion::{
    additive_multi_task_multi, AdditiveMultiTaskConfig, Normalization, validate::validate,
};

/// Product ranking signals for e-commerce.
#[derive(Debug, Clone)]
struct ProductSignals {
    product_id: String,
    ctr: f32,      // Click-through rate
    ctcvr: f32,    // Click-through conversion rate
    revenue: f32,  // Average revenue per click
}

/// E-commerce ranking pipeline using multi-task fusion.
struct EcommerceRankingPipeline;

impl EcommerceRankingPipeline {
    /// Rank products using additive multi-task fusion.
    ///
    /// Combines CTR and CTCVR signals with configurable weights and normalization.
    ///
    /// # Arguments
    /// * `products` - List of products with their signals
    /// * `ctr_weight` - Weight for CTR signal (default: 1.0)
    /// * `ctcvr_weight` - Weight for CTCVR signal (default: 1.0)
    /// * `normalization` - Normalization method (default: MinMax)
    /// * `top_k` - Number of top products to return
    ///
    /// # Returns
    /// Ranked products sorted by fused score (descending)
    pub fn rank_products(
        &self,
        products: &[ProductSignals],
        ctr_weight: f32,
        ctcvr_weight: f32,
        normalization: Normalization,
        top_k: usize,
    ) -> Vec<(String, f32)> {
        println!("\n=== E-commerce Product Ranking ===\n");

        // Convert products to ranked lists
        let ctr_list: Vec<(String, f32)> = products
            .iter()
            .map(|p| (p.product_id.clone(), p.ctr))
            .collect();

        let ctcvr_list: Vec<(String, f32)> = products
            .iter()
            .map(|p| (p.product_id.clone(), p.ctcvr))
            .collect();

        println!("CTR signals (top 5):");
        for (id, score) in ctr_list.iter().take(5) {
            println!("  {}: {:.4}", id, score);
        }

        println!("\nCTCVR signals (top 5):");
        for (id, score) in ctcvr_list.iter().take(5) {
            println!("  {}: {:.4}", id, score);
        }

        // Fuse using additive multi-task fusion
        let config = AdditiveMultiTaskConfig::new((ctr_weight, ctcvr_weight))
            .with_normalization(normalization)
            .with_top_k(top_k);

        // additive_multi_task_multi expects weighted lists: &[(list, weight)]
        let weighted_lists = vec![
            (ctr_list.as_slice(), ctr_weight),
            (ctcvr_list.as_slice(), ctcvr_weight),
        ];

        let fused = additive_multi_task_multi(&weighted_lists, config);

        println!("\nFused ranking (top {}):", top_k);
        for (i, (id, score)) in fused.iter().enumerate() {
            // Find original product for display
            let product = products
                .iter()
                .find(|p| p.product_id == *id)
                .expect("Product not found");
            println!(
                "  {}. {}: score={:.4} (CTR={:.4}, CTCVR={:.4}, Revenue=${:.2})",
                i + 1, id, score, product.ctr, product.ctcvr, product.revenue
            );
        }

        // Validate results
        let validation = validate(&fused, false, Some(top_k));
        if !validation.is_valid {
            eprintln!("⚠️  Validation errors: {:?}", validation.errors);
        } else {
            println!("\n✅ Results validated successfully");
        }

        fused
    }
}

fn main() {
    // Example: Product catalog with CTR and CTCVR signals
    // In production, these would come from analytics/ML models
    let products = vec![
        ProductSignals {
            product_id: "prod_001".to_string(),
            ctr: 0.15,      // 15% click-through rate
            ctcvr: 0.08,    // 8% click-through conversion rate
            revenue: 29.99,
        },
        ProductSignals {
            product_id: "prod_002".to_string(),
            ctr: 0.12,
            ctcvr: 0.12,    // Higher conversion, lower CTR
            revenue: 49.99,
        },
        ProductSignals {
            product_id: "prod_003".to_string(),
            ctr: 0.20,      // High CTR, lower conversion
            ctcvr: 0.05,
            revenue: 19.99,
        },
        ProductSignals {
            product_id: "prod_004".to_string(),
            ctr: 0.10,
            ctcvr: 0.10,
            revenue: 39.99,
        },
        ProductSignals {
            product_id: "prod_005".to_string(),
            ctr: 0.18,
            ctcvr: 0.09,
            revenue: 24.99,
        },
    ];

    let pipeline = EcommerceRankingPipeline;

    // Scenario 1: Equal weights, MinMax normalization (default)
    println!("\n=== Scenario 1: Equal Weights (CTR=1.0, CTCVR=1.0) ===");
    let ranked_1 = pipeline.rank_products(&products, 1.0, 1.0, Normalization::MinMax, 5);

    // Scenario 2: Emphasize conversion (higher CTCVR weight)
    println!("\n=== Scenario 2: Emphasize Conversion (CTR=1.0, CTCVR=2.0) ===");
    let ranked_2 = pipeline.rank_products(&products, 1.0, 2.0, Normalization::MinMax, 5);

    // Scenario 3: Z-score normalization (robust to outliers)
    println!("\n=== Scenario 3: Z-Score Normalization (CTR=1.0, CTCVR=1.0) ===");
    let ranked_3 = pipeline.rank_products(&products, 1.0, 1.0, Normalization::ZScore, 5);

    // Compare rankings
    println!("\n=== Ranking Comparison ===");
    println!("Equal weights top 3: {:?}", ranked_1.iter().take(3).map(|(id, _)| id).collect::<Vec<_>>());
    println!("Conversion-focused top 3: {:?}", ranked_2.iter().take(3).map(|(id, _)| id).collect::<Vec<_>>());
    println!("Z-score top 3: {:?}", ranked_3.iter().take(3).map(|(id, _)| id).collect::<Vec<_>>());

    println!("\n💡 Production considerations:");
    println!("  - A/B test different weight combinations");
    println!("  - Monitor revenue impact of ranking changes");
    println!("  - Use explainability to understand ranking decisions");
    println!("  - Consider adding more signals (revenue, inventory, etc.)");
}

