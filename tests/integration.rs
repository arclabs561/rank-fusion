//! Integration tests simulating realistic e2e workflows.
//!
//! These tests verify rank fusion works correctly in real-world scenarios.

use rank_fusion::{
    borda, combmnz, combmnz_multi, combsum, rrf, rrf_into,
    rrf_multi, weighted, weighted_multi, FusionConfig, RrfConfig, WeightedConfig,
};

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Hybrid Search with BM25 + Dense
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_hybrid_bm25_dense() {
    // Simulated BM25 results (high values)
    let bm25 = vec![
        ("doc_rust", 15.2),
        ("doc_python", 12.1),
        ("doc_go", 8.5),
        ("doc_java", 5.0),
    ];

    // Simulated dense retrieval (cosine similarity)
    let dense = vec![
        ("doc_rust", 0.95),
        ("doc_cpp", 0.88),
        ("doc_python", 0.82),
        ("doc_kotlin", 0.75),
    ];

    // RRF should handle the scale difference
    let fused = rrf(bm25.clone(), dense.clone(), RrfConfig::default());

    // doc_rust appears in both lists at top positions
    assert_eq!(fused[0].0, "doc_rust", "doc_rust should be #1 (in both lists at top)");

    // Should include all unique docs
    let unique_docs: std::collections::HashSet<_> = fused.iter().map(|(id, _)| *id).collect();
    assert_eq!(unique_docs.len(), 6);
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Three-Way Fusion
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_three_way_fusion() {
    let bm25 = vec![("d1", 10.0), ("d2", 8.0), ("d3", 6.0)];
    let dense = vec![("d2", 0.9), ("d4", 0.8), ("d1", 0.7)];
    let keyword = vec![("d1", 5.0), ("d5", 4.0), ("d2", 3.0)];

    // RRF multi
    let fused = rrf_multi(&[&bm25, &dense, &keyword], RrfConfig::default());

    // d1 and d2 appear in all three lists
    let top_2: Vec<_> = fused.iter().take(2).map(|(id, _)| *id).collect();
    assert!(top_2.contains(&"d1") || top_2.contains(&"d2"));

    // CombMNZ multi should also favor overlap
    let combmnz_fused = combmnz_multi(&[&bm25, &dense, &keyword], FusionConfig::default());
    let top_2_mnz: Vec<_> = combmnz_fused.iter().take(2).map(|(id, _)| *id).collect();
    assert!(top_2_mnz.contains(&"d1") || top_2_mnz.contains(&"d2"));
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Weighted Fusion with Tuned Weights
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_weighted_tuned() {
    // Same doc in both lists with different scores
    let retriever_a = vec![("d1", 1.0), ("d2", 0.5)];
    let retriever_b = vec![("d2", 1.0), ("d1", 0.5)];

    // Trust retriever A heavily → d1 should win
    let favor_a = weighted(
        &retriever_a,
        &retriever_b,
        WeightedConfig::new(0.9, 0.1).with_normalize(true),
    );
    assert_eq!(favor_a[0].0, "d1", "Should favor retriever A's ranking");

    // Trust retriever B heavily → d2 should win
    let favor_b = weighted(
        &retriever_a,
        &retriever_b,
        WeightedConfig::new(0.1, 0.9).with_normalize(true),
    );
    assert_eq!(favor_b[0].0, "d2", "Should favor retriever B's ranking");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Buffer Reuse for High Throughput
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_buffer_reuse() {
    let mut output_buffer: Vec<(u32, f32)> = Vec::with_capacity(100);

    // Simulate processing many queries with integer IDs
    for i in 0..10u32 {
        let list_a: Vec<(u32, f32)> = (0..20u32).map(|j| (i * 100 + j, 1.0 - j as f32 * 0.05)).collect();
        let list_b: Vec<(u32, f32)> = (10..30u32).map(|j| (i * 100 + j, 0.9 - (j - 10) as f32 * 0.05)).collect();

        rrf_into(&list_a, &list_b, RrfConfig::default(), &mut output_buffer);

        assert!(!output_buffer.is_empty());
        // Buffer is reused (clear + repopulate)
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: CombMNZ Overlap Bonus
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_combmnz_overlap_bonus() {
    // doc1 appears in both with similar normalized scores
    let list_a = vec![("doc1", 1.0), ("doc2", 0.5)];
    let list_b = vec![("doc1", 1.0), ("doc3", 0.5)];

    let combsum_result = combsum(&list_a, &list_b);
    let combmnz_result = combmnz(&list_a, &list_b);

    // doc1 appears in both, so CombMNZ should give it higher relative score
    let doc1_combsum = combsum_result.iter().find(|(id, _)| *id == "doc1").unwrap().1;
    let doc1_combmnz = combmnz_result.iter().find(|(id, _)| *id == "doc1").unwrap().1;

    // CombMNZ multiplies by overlap count (2), so should be ~2x
    assert!(
        doc1_combmnz > doc1_combsum * 1.5,
        "CombMNZ should boost overlapping docs: {} vs {}",
        doc1_combmnz,
        doc1_combsum
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: RRF k Parameter Tuning
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_rrf_k_tuning() {
    let list_a = vec![("top", 1.0), ("mid", 0.5), ("low", 0.1)];
    let list_b = vec![("other", 0.9)];

    // Low k = top positions dominate
    let low_k = rrf(list_a.clone(), list_b.clone(), RrfConfig::new(1));

    // High k = more uniform contribution
    let high_k = rrf(list_a, list_b, RrfConfig::new(1000));

    // With low k, the spread between scores is larger
    let low_k_spread = low_k[0].1 - low_k[low_k.len() - 1].1;
    let high_k_spread = high_k[0].1 - high_k[high_k.len() - 1].1;

    assert!(
        low_k_spread > high_k_spread,
        "Low k should have larger spread: {} vs {}",
        low_k_spread,
        high_k_spread
    );
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Empty List Handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_empty_lists() {
    let populated = vec![("d1", 1.0), ("d2", 0.5)];
    let empty: Vec<(&str, f32)> = vec![];

    // RRF with one empty list
    let fused = rrf(populated.clone(), empty.clone(), RrfConfig::default());
    assert_eq!(fused.len(), 2, "Should include all docs from non-empty list");

    // Both empty
    let both_empty = rrf(empty.clone(), empty, RrfConfig::default());
    assert!(both_empty.is_empty(), "Empty inputs should produce empty output");

    // CombSUM with empty
    let combsum_result = combsum(&populated, &[]);
    assert_eq!(combsum_result.len(), 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Integer IDs
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_integer_ids() {
    let list_a: Vec<(u64, f32)> = vec![(1001, 0.9), (1002, 0.8), (1003, 0.7)];
    let list_b: Vec<(u64, f32)> = vec![(1002, 0.95), (1004, 0.85), (1001, 0.75)];

    let fused = rrf(list_a, list_b, RrfConfig::default());

    // Verify integer IDs work correctly
    assert!(!fused.is_empty());
    let ids: Vec<_> = fused.iter().map(|(id, _)| *id).collect();
    assert!(ids.contains(&1001));
    assert!(ids.contains(&1002));
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: top_k Filtering
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_top_k_filtering() {
    let list_a: Vec<_> = (0..50).map(|i| (format!("d{}", i), 1.0 - i as f32 * 0.02)).collect();
    let list_b: Vec<_> = (25..75).map(|i| (format!("d{}", i), 0.9 - (i - 25) as f32 * 0.02)).collect();

    let list_a_ref: Vec<_> = list_a.iter().map(|(id, s)| (id.as_str(), *s)).collect();
    let list_b_ref: Vec<_> = list_b.iter().map(|(id, s)| (id.as_str(), *s)).collect();

    // Request top 10
    let fused = rrf(list_a_ref, list_b_ref, RrfConfig::default().with_top_k(10));

    assert_eq!(fused.len(), 10, "Should return exactly top_k results");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Deterministic Output
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_deterministic() {
    let list_a = vec![("d1", 0.9), ("d2", 0.8)];
    let list_b = vec![("d2", 0.9), ("d3", 0.8)];

    // Run multiple times
    let results: Vec<_> = (0..10)
        .map(|_| rrf(list_a.clone(), list_b.clone(), RrfConfig::default()))
        .collect();

    // All results should be identical
    for result in &results[1..] {
        assert_eq!(
            results[0].len(),
            result.len(),
            "Result length should be deterministic"
        );
        for ((id1, score1), (id2, score2)) in results[0].iter().zip(result.iter()) {
            assert_eq!(id1, id2, "IDs should be deterministic");
            assert!((score1 - score2).abs() < 1e-6, "Scores should be deterministic");
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: Borda Count Ordering
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_borda_ordering() {
    // doc1 is #1 in both lists
    let list_a = vec![("doc1", 0.9), ("doc2", 0.8), ("doc3", 0.7)];
    let list_b = vec![("doc1", 0.95), ("doc4", 0.85), ("doc2", 0.75)];

    let fused = borda(&list_a, &list_b);

    // doc1 at position 0 in both lists → highest Borda score
    assert_eq!(fused[0].0, "doc1", "doc1 should be #1 with Borda");

    // Borda scores: doc1 = (3-0) + (3-0) = 6
    //               doc2 = (3-1) + (3-2) = 3
    let doc1_score = fused.iter().find(|(id, _)| *id == "doc1").unwrap().1;
    assert!((doc1_score - 6.0).abs() < 0.01, "doc1 Borda score should be 6");
}

// ─────────────────────────────────────────────────────────────────────────────
// E2E Test: weighted_multi Error Handling
// ─────────────────────────────────────────────────────────────────────────────

#[test]
fn e2e_weighted_multi_errors() {
    let list = vec![("d1", 1.0)];

    // Zero weights should error
    let result = weighted_multi(&[(&list, 0.0)], false, None);
    assert!(result.is_err(), "Zero weights should return error");

    // Valid weights should succeed
    let result = weighted_multi(&[(&list, 1.0), (&list, 1.0)], true, None);
    assert!(result.is_ok(), "Valid weights should succeed");
}

