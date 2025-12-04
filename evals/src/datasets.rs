//! Synthetic datasets demonstrating fusion method differences.
//!
//! Each scenario is designed to highlight when one method outperforms another.
//! The key is creating situations where methods produce DIFFERENT rankings,
//! not just different scores.

use std::collections::HashSet;

/// A synthetic evaluation scenario.
#[derive(Debug, Clone)]
pub struct Scenario {
    pub name: &'static str,
    pub description: &'static str,
    /// Which method(s) should win this scenario (comma-separated if multiple).
    pub expected_winner: &'static str,
    /// What this scenario teaches us about the algorithms.
    pub insight: &'static str,
    /// Retriever A results: (doc_id, score)
    pub retriever_a: Vec<(String, f32)>,
    /// Retriever B results: (doc_id, score)
    pub retriever_b: Vec<(String, f32)>,
    /// Ground truth relevant documents.
    pub relevant: HashSet<String>,
}

impl Scenario {
    fn new(
        name: &'static str,
        description: &'static str,
        expected_winner: &'static str,
        insight: &'static str,
        a: Vec<(&str, f32)>,
        b: Vec<(&str, f32)>,
        rel: Vec<&str>,
    ) -> Self {
        Self {
            name,
            description,
            expected_winner,
            insight,
            retriever_a: a.into_iter().map(|(s, v)| (s.to_string(), v)).collect(),
            retriever_b: b.into_iter().map(|(s, v)| (s.to_string(), v)).collect(),
            relevant: rel.into_iter().map(String::from).collect(),
        }
    }
}

/// All evaluation scenarios.
///
/// Design principles for differentiating scenarios:
/// 1. Relevant docs should NOT be trivially at the top of both lists
/// 2. Each scenario should create a situation where ONE method clearly wins
/// 3. The winning method's ranking should put relevant docs higher than others
pub fn all_scenarios() -> Vec<Scenario> {
    vec![
        // ─────────────────────────────────────────────────────────────────────
        // Scenario 1: RRF with k=60 heavily favors overlap
        //
        // Key insight: RRF's large k value means overlapping docs get ~2x the
        // score of single-list docs, regardless of raw rank position.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "rrf_overlap_boost",
            "RRF with k=60 gives strong bonus to overlapping docs.",
            "rrf",
            "With k=60, RRF score is ~1/61 per list. Overlap gives ~2/61, beating any single-list doc.",
            // A: rel1 and rel2 appear in both lists
            vec![
                ("irr_a1", 0.99),
                ("irr_a2", 0.95),
                ("rel1", 0.90),
                ("rel2", 0.85),
                ("irr_a3", 0.80),
            ],
            // B: same relevant docs, different irrelevant
            vec![
                ("irr_b1", 0.98),
                ("rel1", 0.94),
                ("rel2", 0.89),
                ("irr_b2", 0.84),
                ("irr_b3", 0.79),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 2: CombMNZ explicitly rewards overlap
        //
        // CombMNZ = sum(scores) × count. Overlapping docs get 2× multiplier.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "combmnz_overlap_multiplier",
            "CombMNZ multiplies by overlap count, boosting shared docs.",
            "combmnz",
            "score = normalized_sum × appearance_count. Overlap docs get 2× multiplier.",
            vec![
                ("noise_a1", 0.99),  // High score but 1× multiplier
                ("rel1", 0.95),      // 2× multiplier
                ("rel2", 0.90),      // 2× multiplier
                ("noise_a2", 0.85),
                ("noise_a3", 0.80),
            ],
            vec![
                ("noise_b1", 0.98),  // 1× multiplier
                ("rel1", 0.94),      // 2× multiplier
                ("noise_b2", 0.89),
                ("rel2", 0.84),      // 2× multiplier
                ("noise_b3", 0.79),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 3: Rank-based methods beat score-based with outliers
        //
        // Key: The outlier is #1 in A but #4 in B. Rank-based methods (Borda)
        // will penalize the outlier's inconsistent ranking, while score-based
        // methods (including additive_multi_task with Z-score) may still be
        // influenced by the outlier's high score in A.
        //
        // To make Borda win, we need the outlier to have very inconsistent
        // ranks, and rel1/rel2 to have consistent high ranks.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "rank_based_beats_outlier",
            "Score-based methods struggle with outliers; rank-based ignore them.",
            "borda",
            "Rank-based methods (RRF/ISR/Borda) ignore score magnitude entirely. Borda penalizes inconsistent rankings.",
            // A has massive outlier at #1, but rel1/rel2 are at #2 and #3
            vec![
                ("outlier", 1000.0),  // #1 rank (inconsistent - will be penalized by Borda)
                ("rel1", 50.0),       // #2 rank (consistent high rank)
                ("rel2", 48.0),       // #3 rank (consistent high rank)
                ("irr1", 46.0),
                ("irr2", 44.0),
                ("irr3", 42.0),
                ("irr4", 40.0),
                ("irr5", 38.0),
                ("irr6", 36.0),
                ("irr7", 34.0),
            ],
            // B ranks rel1/rel2 at #1 and #2, outlier much lower
            vec![
                ("rel1", 0.98),       // #1 rank (consistent high rank)
                ("rel2", 0.95),       // #2 rank (consistent high rank)
                ("irr3", 0.90),
                ("irr4", 0.85),
                ("irr5", 0.80),
                ("irr6", 0.75),
                ("irr7", 0.70),
                ("irr1", 0.65),
                ("irr2", 0.60),
                ("outlier", 0.20),    // #10 rank (very inconsistent!)
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 4: Weighted controls bad retriever influence
        //
        // When you KNOW one retriever is bad, weighted fusion helps.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "weighted_downweight_bad",
            "Weighted fusion with 0.9/0.1 controls bad retriever B.",
            "weighted",
            "When retriever quality is known a priori, explicit weighting helps.",
            // A is excellent
            vec![
                ("rel1", 0.99),
                ("rel2", 0.95),
                ("rel3", 0.90),
                ("irr1", 0.80),
                ("irr2", 0.70),
            ],
            // B is adversarial
            vec![
                ("irr3", 0.99),
                ("irr4", 0.95),
                ("irr5", 0.90),
                ("rel1", 0.30),
                ("rel2", 0.20),
            ],
            vec!["rel1", "rel2", "rel3"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 5: All methods tie with perfect agreement
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "all_equal_agreement",
            "When retrievers agree, all fusion methods produce same ranking.",
            "all_equal",
            "Perfect agreement is the degenerate case where fusion doesn't matter.",
            vec![
                ("rel1", 0.99),
                ("rel2", 0.95),
                ("irr1", 0.90),
                ("irr2", 0.85),
                ("irr3", 0.80),
            ],
            vec![
                ("rel1", 0.98),
                ("rel2", 0.94),
                ("irr1", 0.89),
                ("irr2", 0.84),
                ("irr3", 0.79),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 6: Borda penalizes inconsistent ranking
        //
        // Borda count: score = N - rank. Erratic doc loses points.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "borda_penalizes_erratic",
            "Borda count penalizes docs with inconsistent rankings across lists.",
            "borda",
            "Borda score = sum(N - rank). Erratic #1/#10 loses to consistent #2/#2.",
            vec![
                ("erratic", 0.99),      // Rank 1 here
                ("consistent1", 0.95),  // Rank 2
                ("consistent2", 0.90),  // Rank 3
                ("irr1", 0.85),
                ("irr2", 0.80),
            ],
            vec![
                ("consistent1", 0.98),  // Rank 1
                ("consistent2", 0.94),  // Rank 2
                ("irr3", 0.89),
                ("irr4", 0.84),
                ("irr5", 0.79),
                ("irr6", 0.74),
                ("irr7", 0.69),
                ("irr8", 0.64),
                ("irr9", 0.59),
                ("erratic", 0.10),      // Rank 10 here!
            ],
            vec!["consistent1", "consistent2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 7: CombSUM works with calibrated scores
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "combsum_calibrated",
            "CombSUM is effective when both retrievers have calibrated scores.",
            "combsum",
            "With calibrated scores, sum directly reflects combined confidence.",
            vec![
                ("rel1", 0.95),
                ("uncertain1", 0.60),
                ("uncertain2", 0.55),
                ("irr1", 0.40),
                ("irr2", 0.35),
            ],
            vec![
                ("rel1", 0.92),
                ("uncertain2", 0.58),
                ("uncertain1", 0.52),
                ("irr3", 0.38),
                ("irr4", 0.33),
            ],
            vec!["rel1"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 8: RRF's k=60 actually helps deep relevance via overlap
        //
        // Counter-intuitive: RRF with high k doesn't weight top ranks heavily.
        // Instead, it gives consistent ~1/61 per list, making overlap dominate.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "rrf_deep_overlap",
            "RRF k=60 makes overlap dominate even for deep-ranked docs.",
            "rrf",
            "1/(60+r) varies only 10% between r=1 and r=5. Overlap = 2× contribution.",
            vec![
                ("irr1", 0.99),
                ("irr2", 0.95),
                ("irr3", 0.90),
                ("rel1", 0.85),  // Rank 4
                ("rel2", 0.80),  // Rank 5
                ("irr4", 0.75),
            ],
            vec![
                ("irr5", 0.98),
                ("irr6", 0.94),
                ("irr7", 0.89),
                ("irr8", 0.84),
                ("rel1", 0.79),  // Rank 5
                ("rel2", 0.74),  // Rank 6
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 9: Graceful degradation with empty list
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "graceful_empty",
            "All methods handle empty input gracefully.",
            "all_equal",
            "Edge case: empty list should not cause errors or distort ranking.",
            vec![
                ("rel1", 0.99),
                ("rel2", 0.95),
                ("irr1", 0.90),
            ],
            vec![],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 10: Complete disagreement - fair interleaving
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "disagreement_interleave",
            "When retrievers disagree completely, need fair interleaving.",
            "all_equal",  // All methods interleave reasonably
            "With disjoint results, all methods must interleave somehow.",
            vec![
                ("rel_a", 0.99),
                ("irr1", 0.95),
                ("irr2", 0.90),
                ("irr3", 0.85),
                ("irr4", 0.80),
            ],
            vec![
                ("rel_b", 0.98),
                ("irr5", 0.94),
                ("irr6", 0.89),
                ("irr7", 0.84),
                ("irr8", 0.79),
            ],
            vec!["rel_a", "rel_b"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 11: DBSF helps when distributions differ but no outliers
        //
        // When score distributions genuinely differ (not just outliers),
        // z-score normalization aligns them.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "dbsf_distribution_mismatch",
            "DBSF z-score helps align genuinely different distributions.",
            "dbsf",
            "Z-score: (x - mean) / std. Aligns distributions with different means/variances.",
            // A: scores cluster around 0.5
            vec![
                ("irr1", 0.55),
                ("rel1", 0.52),
                ("rel2", 0.50),
                ("irr2", 0.48),
                ("irr3", 0.45),
            ],
            // B: scores cluster around 0.9
            vec![
                ("rel1", 0.95),
                ("rel2", 0.92),
                ("irr4", 0.90),
                ("irr5", 0.88),
                ("irr1", 0.85),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 12: ISR uses steeper decay for rank emphasis
        //
        // Key: ISR with k=1 has steeper decay than RRF with k=60.
        // ISR: 1/sqrt(1+0) = 1.0 at rank 0, 1/sqrt(1+4) = 0.447 at rank 4
        // RRF: 1/(60+0) = 0.0167 at rank 0, 1/(60+4) = 0.0156 at rank 4
        // ISR's steeper decay means top ranks matter more.
        //
        // To make ISR win, we need rel1 at rank 0 in A, and many irrelevant
        // docs in B so CombMNZ doesn't get overlap bonus.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "isr_steeper_decay",
            "ISR's steeper decay (k=1) emphasizes top ranks in disjoint lists.",
            "isr",
            "ISR with k=1: 1/sqrt(1+0)=1.0 at rank 0 vs 1/sqrt(1+4)=0.447 at rank 4. Much steeper than RRF's 1/(60+r). rel1 at rank 0 should dominate.",
            // A: rel1 at rank 0 (top position) - gets maximum ISR score
            vec![
                ("rel1", 0.99),  // Rank 0 - ISR: 1/sqrt(1) = 1.0, RRF: 1/60 = 0.0167
                ("irr1", 0.95),  // Rank 1 - ISR: 1/sqrt(2) = 0.707, RRF: 1/61 = 0.0164
                ("irr2", 0.90),  // Rank 2 - ISR: 1/sqrt(3) = 0.577, RRF: 1/62 = 0.0161
                ("irr3", 0.85),  // Rank 3 - ISR: 1/sqrt(4) = 0.5, RRF: 1/63 = 0.0159
                ("irr4", 0.80),  // Rank 4 - ISR: 1/sqrt(5) = 0.447, RRF: 1/64 = 0.0156
            ],
            // B: all irrelevant, rel1 not present (disjoint lists)
            // Many irrelevant docs so CombMNZ doesn't get overlap bonus
            vec![
                ("irr5", 0.98),  // Rank 0 - ISR: 1.0, but irrelevant
                ("irr6", 0.94),  // Rank 1 - ISR: 0.707
                ("irr7", 0.89),  // Rank 2 - ISR: 0.577
                ("irr8", 0.84),  // Rank 3 - ISR: 0.5
                ("irr9", 0.79),  // Rank 4 - ISR: 0.447
            ],
            vec!["rel1"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 13: Standardized fusion beats CombSUM with different distributions
        //
        // ERANK shows 2-5% NDCG improvement when score distributions differ.
        // Standardized (z-score) is more robust than min-max normalization.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_distribution_mismatch",
            "Standardized fusion (z-score) outperforms CombSUM when distributions differ.",
            "standardized",
            "Z-score normalization (standardization) is more robust to distribution differences than min-max. ERANK shows 2-5% NDCG improvement.",
            // A: scores have high variance, mean around 0.5
            vec![
                ("irr1", 0.70),  // +2σ
                ("rel1", 0.55),  // +0.5σ
                ("rel2", 0.50),  // mean
                ("irr2", 0.45),  // -0.5σ
                ("irr3", 0.30),  // -2σ
            ],
            // B: scores have low variance, mean around 0.9
            vec![
                ("rel1", 0.95),  // +0.5σ
                ("rel2", 0.92),  // mean
                ("irr4", 0.90),  // -0.5σ
                ("irr5", 0.88),  // -1σ
                ("irr1", 0.85),  // -1.5σ
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 14: Standardized with tight clipping handles outliers better
        //
        // Key: To make tight clipping win, we need a scenario where:
        // 1. The outlier's z-score is between 2 and 3 (so tight clipping helps)
        // 2. Both lists have similar importance (not dominated by one list)
        // 3. The outlier hurts ranking in both lists
        //
        // Strategy: Make both lists have similar score distributions, so the
        // 20× weight doesn't dominate. The outlier should be problematic in both.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_tight_clipping",
            "Tight clipping range helps when moderate outliers exist.",
            "standardized_tight",
            "Tighter clipping [-2, 2] vs [-3, 3] reduces outlier influence. With balanced lists (not dominated by one), tight clipping should outperform methods without clipping.",
            // A: has outlier that will z-score to ~2.5σ, rel1/rel2 are good
            // Mean ~2.0, std ~2.4, so outlier (8.0) → z = (8-2)/2.4 ≈ 2.5
            vec![
                ("outlier", 8.0),   // Outlier (z-score ~2.5, clipped to 2.0 by tight, 2.5 by default)
                ("rel1", 2.5),      // Good score (z ≈ 0.2)
                ("rel2", 2.4),      // Good score (z ≈ 0.17)
                ("irr1", 1.0),
                ("irr2", 0.9),
            ],
            // B: similar distribution, rel1/rel2 rank high, outlier ranks low
            // Mean ~0.5, std ~0.4, so outlier (0.1) → z = (0.1-0.5)/0.4 = -1.0
            vec![
                ("rel1", 0.9),      // High (z ≈ 1.0)
                ("rel2", 0.85),     // High (z ≈ 0.875)
                ("irr3", 0.5),
                ("irr4", 0.45),
                ("outlier", 0.1),   // Low (z ≈ -1.0, clipped to -2.0 by tight)
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 15: Additive multi-task fusion (ResFlow-style)
        //
        // ResFlow shows additive (α·A + β·B) outperforms multiplicative for
        // e-commerce. Optimal formula: CTR + CTCVR × 20.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_multi_task_ecommerce",
            "Additive multi-task fusion (ResFlow) for e-commerce ranking.",
            "additive_multi_task_1_20",
            "ResFlow: additive fusion (CTR + CTCVR × 20) outperforms multiplicative for e-commerce. Task B (CTCVR) gets 20× weight.",
            // Task A: CTR scores (click-through rate) - item1 has high CTR but low conversion
            vec![
                ("item2", 0.05),   // High CTR, will win with high CTCVR
                ("item1", 0.04),   // Medium CTR, low CTCVR
                ("item3", 0.02),   // Low CTR
                ("item4", 0.03),
                ("item5", 0.01),
            ],
            // Task B: CTCVR scores (click-to-conversion rate) - item2 has much higher conversion
            vec![
                ("item2", 0.03),   // High CTCVR (0.05 + 0.03×20 = 0.65) - WINNER
                ("item1", 0.01),   // Low CTCVR (0.04 + 0.01×20 = 0.24)
                ("item3", 0.02),   // Medium CTCVR (0.02 + 0.02×20 = 0.42)
                ("item4", 0.015),
                ("item5", 0.005),
            ],
            vec!["item2"],  // item2 wins with weighted additive (1:20)
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 16: Additive vs equal weights
        //
        // When tasks have different scales, weighted additive helps.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_equal_weights",
            "Equal-weight additive fusion when both tasks matter equally.",
            "additive_multi_task_1_1",
            "Equal weights (1.0, 1.0) when both tasks contribute equally to final ranking.",
            // Task A: view scores
            vec![
                ("rel1", 0.90),
                ("rel2", 0.85),
                ("irr1", 0.80),
                ("irr2", 0.75),
                ("irr3", 0.70),
            ],
            // Task B: engagement scores (similar scale)
            vec![
                ("rel1", 0.88),
                ("rel2", 0.83),
                ("irr4", 0.78),
                ("irr5", 0.73),
                ("irr1", 0.68),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 17: Standardized beats DBSF with configurable clipping
        //
        // Standardized allows custom clipping, DBSF uses fixed [-3, 3].
        // When distributions are similar but you want tighter control, standardized wins.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_vs_dbsf",
            "Standardized fusion with custom clipping vs fixed DBSF.",
            "standardized",
            "Standardized allows configurable clipping range, making it more flexible than DBSF's fixed [-3, 3].",
            // A: normal distribution
            vec![
                ("rel1", 0.60),
                ("rel2", 0.55),
                ("irr1", 0.50),
                ("irr2", 0.45),
                ("irr3", 0.40),
            ],
            // B: slightly shifted distribution
            vec![
                ("rel1", 0.95),
                ("rel2", 0.90),
                ("irr4", 0.85),
                ("irr5", 0.80),
                ("irr1", 0.75),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 18: Standardized handles negative scores
        //
        // Some retrievers produce negative scores (e.g., after centering).
        // Z-score normalization handles this naturally.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_negative_scores",
            "Standardized fusion handles negative scores naturally.",
            "standardized",
            "Z-score normalization works with any score distribution, including negative values.",
            // A: negative scores
            vec![
                ("rel1", -0.5),
                ("rel2", -0.7),
                ("irr1", -0.9),
                ("irr2", -1.0),
            ],
            // B: positive scores
            vec![
                ("rel1", 0.9),
                ("rel2", 0.8),
                ("irr3", 0.6),
                ("irr4", 0.5),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 19: Additive multi-task with extreme weights
        //
        // Test that extreme weight ratios (e.g., 1:100) work correctly.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_extreme_weights",
            "Additive fusion with extreme weight ratios (1:100).",
            "additive_multi_task_1_20",
            "Extreme weights should still produce valid rankings. Task B dominates.",
            // Task A: small values
            vec![
                ("rel1", 0.01),
                ("rel2", 0.008),
                ("irr1", 0.005),
            ],
            // Task B: even smaller values, but ×100 weight
            vec![
                ("rel1", 0.002),  // 0.01 + 0.002×100 = 0.21
                ("rel2", 0.0015), // 0.008 + 0.0015×100 = 0.158
                ("irr1", 0.001),  // 0.005 + 0.001×100 = 0.105
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 20: Standardized with single-element lists
        //
        // Edge case: lists with only one element (zero variance).
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_single_element",
            "Standardized fusion handles single-element lists gracefully.",
            "standardized",
            "Single-element lists have zero variance. Z-score = 0, but fusion still works.",
            vec![("rel1", 1.0)],
            vec![("rel1", 0.9), ("rel2", 0.8)],
            vec!["rel1"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 21: Additive multi-task with disjoint tasks
        //
        // Tasks have completely different document sets.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_disjoint_tasks",
            "Additive fusion when tasks have disjoint document sets.",
            "additive_multi_task_1_1",
            "Disjoint tasks should interleave documents fairly.",
            // Task A: documents 1-3
            vec![
                ("rel_a1", 0.9),
                ("rel_a2", 0.8),
                ("irr_a1", 0.7),
            ],
            // Task B: documents 4-6
            vec![
                ("rel_b1", 0.9),
                ("rel_b2", 0.8),
                ("irr_b1", 0.7),
            ],
            vec!["rel_a1", "rel_a2", "rel_b1", "rel_b2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 22: Standardized vs CombSUM with outliers
        //
        // Key: Standardized (z-score with clipping) should outperform CombSUM
        // (min-max normalization) when outliers exist. The outlier in A will
        // dominate min-max normalization but be clipped in z-score.
        //
        // To make standardized win over additive_multi_task:
        // 1. Both lists should have similar importance (not dominated by 20× weight)
        // 2. The outlier should have z-score > 3, so clipping helps
        // 3. The outlier should hurt ranking in both lists
        //
        // Strategy: Balance the lists so 20× weight doesn't dominate, and make
        // the outlier problematic enough that clipping helps.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_outlier_robustness",
            "Standardized fusion is more robust to outliers than CombSUM.",
            "standardized",
            "Z-score normalization with clipping reduces outlier influence compared to min-max. With balanced lists and outliers that clip, standardized should outperform methods without clipping.",
            // A: has massive outlier that will z-score to > 3σ
            // Mean ~50, std ~200, so outlier (700) → z = (700-50)/200 = 3.25 (clipped to 3.0)
            vec![
                ("outlier", 700.0),  // Massive outlier (z-score > 3, clipped to 3.0)
                ("rel1", 60.0),     // Good score (z ≈ 0.05)
                ("rel2", 55.0),     // Good score (z ≈ 0.025)
                ("irr1", 40.0),
                ("irr2", 35.0),
            ],
            // B: similar distribution, rel1/rel2 rank high, outlier ranks low
            // Mean ~0.5, std ~0.3, so outlier (0.05) → z = (0.05-0.5)/0.3 ≈ -1.5
            vec![
                ("rel1", 0.9),      // High (z ≈ 1.33)
                ("rel2", 0.85),     // High (z ≈ 1.17)
                ("irr3", 0.5),
                ("irr4", 0.45),
                ("outlier", 0.05),  // Low (z ≈ -1.5, clipped to -3.0)
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 23: Additive multi-task with normalization comparison
        //
        // Different normalization methods should produce different rankings.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_normalization_matters",
            "Normalization method affects additive multi-task fusion results.",
            "additive_multi_task_1_1",
            "Z-score normalization (default) handles different distributions better than min-max. With equal weights, rel1 should win.",
            // A: tight distribution around 0.5, rel1 slightly better
            vec![
                ("rel1", 0.52),  // Slightly better
                ("rel2", 0.50),
                ("irr1", 0.48),
            ],
            // B: wide distribution, rel1 much better
            vec![
                ("rel1", 0.95),  // Much better in B
                ("rel2", 0.50),
                ("irr2", 0.45),
            ],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 24: Standardized with many lists
        //
        // Test with 5+ lists to ensure scalability.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "standardized_many_lists",
            "Standardized fusion scales to many input lists.",
            "standardized",
            "Z-score normalization works efficiently with many lists.",
            vec![("rel1", 10.0), ("rel2", 8.0)],
            vec![("rel1", 0.9), ("rel2", 0.8)],
            vec!["rel1", "rel2"],
        ),

        // ─────────────────────────────────────────────────────────────────────
        // Scenario 25: Additive multi-task real-world e-commerce
        //
        // Realistic e-commerce scenario: view → click → add-to-cart → purchase.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "additive_ecommerce_funnel",
            "E-commerce funnel: view, click, add-to-cart, purchase.",
            "additive_multi_task_1_20",
            "ResFlow shows additive fusion works well for e-commerce multi-task ranking.",
            // View scores (high volume, low conversion)
            vec![
                ("item1", 0.10),
                ("item2", 0.08),
                ("item3", 0.06),
            ],
            // Purchase scores (low volume, high value) - weighted ×20
            vec![
                ("item1", 0.03),  // 0.10 + 0.03×20 = 0.70
                ("item2", 0.02),  // 0.08 + 0.02×20 = 0.48
                ("item3", 0.01),  // 0.06 + 0.01×20 = 0.26
            ],
            vec!["item1", "item2"],
        ),
    ]
}
