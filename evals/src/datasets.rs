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
        // DBSF actually struggles here because z-score amplifies the outlier.
        // Rank-based methods (RRF, ISR, Borda) ignore scores entirely.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "rank_based_beats_outlier",
            "Score-based methods struggle with outliers; rank-based ignore them.",
            "borda",
            "Rank-based methods (RRF/ISR/Borda) ignore score magnitude entirely.",
            // A has massive outlier - but it's #1 in ranks
            vec![
                ("outlier", 1000.0),  // #1 rank
                ("rel1", 50.0),       // #2 rank
                ("rel2", 48.0),       // #3 rank
                ("irr1", 46.0),
                ("irr2", 44.0),
            ],
            // B ranks rel1/rel2 higher than outlier
            vec![
                ("rel1", 0.98),       // #1 rank
                ("rel2", 0.95),       // #2 rank
                ("irr3", 0.90),
                ("outlier", 0.20),    // #4 rank
                ("irr4", 0.15),
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
        // With disjoint lists, rank-based methods need to differentiate.
        // ISR's 1/sqrt(k+r) decay is steeper than RRF's 1/(k+r) at small k.
        // ─────────────────────────────────────────────────────────────────────
        Scenario::new(
            "isr_steeper_decay",
            "ISR's steeper decay (k=1) emphasizes top ranks in disjoint lists.",
            "isr",
            "ISR with k=1: 1/sqrt(2)=0.71 at r=1 vs 1/sqrt(6)=0.41 at r=5. Steeper than RRF.",
            // A: rel1 at rank 1 with highest score
            vec![
                ("rel1", 0.99),
                ("irr1", 0.95),
                ("irr2", 0.90),
                ("irr3", 0.85),
                ("irr4", 0.80),
            ],
            // B: all irrelevant with slightly lower scores
            vec![
                ("irr5", 0.98),
                ("irr6", 0.94),
                ("irr7", 0.89),
                ("irr8", 0.84),
                ("irr9", 0.79),
            ],
            vec!["rel1"],
        ),
    ]
}

