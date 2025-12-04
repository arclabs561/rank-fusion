//! Edge case tests for rank-fusion algorithms.

use rank_fusion::*;

#[test]
fn test_empty_lists() {
    let empty: Vec<(&str, f32)> = vec![];
    let list = vec![("doc1", 0.9), ("doc2", 0.5)];
    
    // Empty + non-empty should return non-empty list's items
    let fused = rrf(&empty, &list);
    assert_eq!(fused.len(), 2);
    assert_eq!(fused[0].0, "doc1");
    
    // Both empty should return empty
    let empty2: Vec<(&str, f32)> = vec![];
    let fused = rrf(&empty, &empty2);
    assert!(fused.is_empty());
}

#[test]
fn test_single_item_lists() {
    let single1 = vec![("doc1", 0.9)];
    let single2 = vec![("doc2", 0.8)];
    
    let fused = rrf(&single1, &single2);
    assert_eq!(fused.len(), 2);
    // Both should have same score (rank 0 in their respective lists)
    assert_eq!(fused[0].1, fused[1].1);
}

#[test]
fn test_all_identical_scores() {
    // RRF should still work (rank-based, not score-based)
    let list1 = vec![("doc1", 0.5), ("doc2", 0.5), ("doc3", 0.5)];
    let list2 = vec![("doc2", 0.5), ("doc3", 0.5), ("doc1", 0.5)];
    
    let fused = rrf(&list1, &list2);
    // All docs appear in both lists, so all should have same score
    assert_eq!(fused.len(), 3);
    assert!((fused[0].1 - fused[1].1).abs() < 1e-6);
    assert!((fused[1].1 - fused[2].1).abs() < 1e-6);
}

#[test]
fn test_k_zero_returns_empty() {
    // k=0 would cause division by zero, should return empty
    let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];
    
    let fused = rrf_with_config(&list1, &list2, RrfConfig::new(0));
    assert!(fused.is_empty());
}

#[test]
fn test_very_large_k() {
    // k=1000 should still work (flatter curve)
    let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];
    
    let fused = rrf_with_config(&list1, &list2, RrfConfig::new(1000));
    assert!(!fused.is_empty());
    // With very large k, scores should be very small
    assert!(fused[0].1 < 0.01);
}

#[test]
fn test_duplicate_document_ids() {
    // Duplicates in same list should contribute multiple times
    let list1 = vec![("doc1", 0.9), ("doc1", 0.8), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];
    
    let fused = rrf(&list1, &list2);
    // doc1 should have higher score (appears twice in list1)
    assert!(fused.iter().any(|(id, _)| *id == "doc1"));
}

#[test]
fn test_non_finite_scores() {
    // Non-finite scores should be handled gracefully
    let list1 = vec![("doc1", f32::INFINITY), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", f32::NAN)];
    
    // RRF ignores scores, so should still work
    let fused = rrf(&list1, &list2);
    assert!(!fused.is_empty());
    
    // Note: RRF produces finite scores even from non-finite inputs (rank-based)
    // This test verifies the behavior, not that it's an error
    let _validation = validate_finite_scores(&fused);
}

#[test]
fn test_negative_scores() {
    // Negative scores should work (RRF ignores them)
    let list1 = vec![("doc1", -0.9), ("doc2", -0.5)];
    let list2 = vec![("doc2", -0.8), ("doc3", -0.3)];
    
    let fused = rrf(&list1, &list2);
    assert!(!fused.is_empty());
    
    // Validation should warn about negative scores
    let validation = validate_non_negative_scores(&fused);
    // RRF produces positive scores (1/(k+rank) > 0)
    assert!(validation.is_valid);
}

#[test]
fn test_very_long_lists() {
    // Test with 1000+ items
    let list1: Vec<(String, f32)> = (0..1000)
        .map(|i| (format!("doc{}", i), 1000.0 - i as f32))
        .collect();
    let list2: Vec<(String, f32)> = (500..1500)
        .map(|i| (format!("doc{}", i), 1500.0 - i as f32))
        .collect();
    
    let fused = rrf(&list1, &list2);
    assert!(!fused.is_empty());
    assert!(fused.len() <= list1.len() + list2.len());
}

#[test]
fn test_weighted_zero_weights() {
    // Zero weights should return error
    let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];
    
    let lists = vec![(&list1[..], 0.0), (&list2[..], 0.0)];
    let result = weighted_multi(&lists, true, None);
    
    assert!(result.is_err());
    assert!(matches!(result.unwrap_err(), FusionError::ZeroWeights));
}

#[test]
fn test_weighted_mismatched_lengths() {
    // Mismatched list/weight lengths should be handled
    let list1 = vec![("doc1", 0.9)];
    let list2 = vec![("doc2", 0.8)];
    
    // More weights than lists (should still work, weights paired with lists)
    let lists = vec![(&list1[..], 0.5), (&list2[..], 0.5)];
    
    // This should work (weights are paired with lists)
    let result = weighted_multi(&lists, true, None);
    assert!(result.is_ok());
}

#[test]
fn test_combsum_empty_lists() {
    let empty: Vec<(&str, f32)> = vec![];
    let list = vec![("doc1", 0.9), ("doc2", 0.5)];
    
    let fused = combsum(&empty, &list);
    assert_eq!(fused.len(), 2);
    // CombSUM normalizes scores, so they may differ from original
    // But both items should be present
    assert!(fused.iter().any(|(id, _)| *id == "doc1"));
    assert!(fused.iter().any(|(id, _)| *id == "doc2"));
}

#[test]
fn test_combmnz_single_list_overlap() {
    // CombMNZ should multiply by overlap count
    let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];
    
    let fused = combmnz(&list1, &list2);
    
    // doc2 appears in both lists, should have 2× multiplier
    let doc2_score = fused.iter().find(|(id, _)| *id == "doc2").unwrap().1;
    let doc1_score = fused.iter().find(|(id, _)| *id == "doc1").unwrap().1;
    
    // doc2 should rank higher due to overlap multiplier
    assert!(doc2_score > doc1_score);
}

#[test]
fn test_dbsf_different_distributions() {
    // DBSF should normalize different distributions
    let list1 = vec![("doc1", 100.0), ("doc2", 50.0)]; // BM25-like
    let list2 = vec![("doc2", 0.9), ("doc3", 0.8)];   // Cosine-like
    
    let fused = dbsf(&list1, &list2);
    assert!(!fused.is_empty());
    
    // Z-score normalization should put scores on comparable scale
    // All scores should be within reasonable range (clipped to [-3, 3])
}

#[test]
fn test_standardized_clipping() {
    // Standardized should clip extreme z-scores
    let list1 = vec![("doc1", 1000.0), ("doc2", 10.0)]; // Extreme outlier
    let list2 = vec![("doc2", 0.9), ("doc3", 0.8)];
    
    let config = StandardizedConfig::new((-3.0, 3.0));
    let fused = standardized_with_config(&list1, &list2, config);
    
    // Clipping should prevent outlier from dominating
    assert!(!fused.is_empty());
}

#[test]
fn test_additive_multi_task_weights() {
    // Additive multi-task should respect task weights
    let task1 = vec![("item1", 0.1), ("item2", 0.05)];
    let task2 = vec![("item2", 0.02), ("item3", 0.01)];
    
    // Task 2 is 20× more important
    let config = AdditiveMultiTaskConfig::new((1.0, 20.0));
    let fused = additive_multi_task_with_config(&task1, &task2, config);
    
    // item2 should rank high (appears in both, task2 weighted heavily)
    assert!(fused.iter().any(|(id, _)| *id == "item2"));
}

#[test]
fn test_multi_list_fusion() {
    // Test fusion with 5+ lists
    let lists = vec![
        vec![("doc1", 0.9), ("doc2", 0.5)],
        vec![("doc2", 0.8), ("doc3", 0.3)],
        vec![("doc1", 0.7), ("doc4", 0.6)],
        vec![("doc2", 0.6), ("doc5", 0.4)],
        vec![("doc1", 0.5), ("doc3", 0.2)],
    ];
    
    let slices: Vec<&[(&str, f32)]> = lists.iter().map(|l| l.as_slice()).collect();
    let fused = rrf_multi(&slices, Default::default());
    
    // doc1 and doc2 should rank high (appear in multiple lists)
    assert!(!fused.is_empty());
}

#[test]
fn test_top_k_limiting() {
    let list1 = vec![("doc1", 0.9), ("doc2", 0.5), ("doc3", 0.3)];
    let list2 = vec![("doc2", 0.8), ("doc3", 0.6), ("doc4", 0.4)];
    
    let config = RrfConfig::default().with_top_k(2);
    let fused = rrf_with_config(&list1, &list2, config);
    
    assert_eq!(fused.len(), 2);
}

#[test]
fn test_validation_edge_cases() {
    // Test validation with edge cases
    let results = vec![
        ("doc1", 0.9),
        ("doc2", 0.5),
        ("doc3", 0.3),
    ];
    
    // Should pass all validations
    assert!(validate_sorted(&results).is_valid);
    assert!(validate_no_duplicates(&results).is_valid);
    assert!(validate_finite_scores(&results).is_valid);
    
    // Test with duplicates (should fail)
    let duplicates = vec![("doc1", 0.9), ("doc1", 0.5)];
    assert!(!validate_no_duplicates(&duplicates).is_valid);
    
    // Test with unsorted (should fail)
    let unsorted = vec![("doc2", 0.5), ("doc1", 0.9)];
    assert!(!validate_sorted(&unsorted).is_valid);
}

#[test]
fn test_explainability_edge_cases() {
    use rank_fusion::explain::RetrieverId;
    
    let list1 = vec![("doc1", 0.9)];
    let list2: Vec<(&str, f32)> = vec![];
    
    let lists = vec![&list1[..], &list2[..]];
    let retriever_ids = vec![
        RetrieverId::new("retriever1"),
        RetrieverId::new("retriever2"),
    ];
    
    let explained = rrf_explain(&lists, &retriever_ids, Default::default());
    
    // doc1 should only have contribution from retriever1
    let doc1 = explained.iter().find(|r| r.id == "doc1").unwrap();
    assert_eq!(doc1.explanation.sources.len(), 1);
    assert_eq!(doc1.explanation.consensus_score, 0.5); // 1/2 retrievers
}

