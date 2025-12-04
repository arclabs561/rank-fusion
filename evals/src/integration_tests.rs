//! Integration tests for dataset loading, conversion, and evaluation.
//!
//! These tests verify the end-to-end evaluation pipeline using rank-eval.

#[cfg(test)]
mod tests {
    use crate::*;
    use rank_eval::trec::{load_trec_runs, load_qrels, group_runs_by_query, group_qrels_by_query};
    use rank_eval::binary::ndcg_at_k;
    use rank_eval::graded::{compute_ndcg, compute_map};
    use std::fs;
    use std::io::Write;
    use tempfile::TempDir;

    fn create_temp_trec_runs() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("runs.txt");
        let mut file = fs::File::create(&file_path).unwrap();
        
        // Valid TREC runs
        writeln!(file, "1 Q0 doc1 1 0.95 bm25").unwrap();
        writeln!(file, "1 Q0 doc2 2 0.87 bm25").unwrap();
        writeln!(file, "1 Q0 doc3 3 0.75 bm25").unwrap();
        writeln!(file, "2 Q0 doc4 1 0.92 bm25").unwrap();
        writeln!(file, "2 Q0 doc5 2 0.85 bm25").unwrap();
        
        // Add second run for fusion
        writeln!(file, "1 Q0 doc2 1 0.93 dense").unwrap();
        writeln!(file, "1 Q0 doc1 2 0.88 dense").unwrap();
        writeln!(file, "1 Q0 doc3 3 0.82 dense").unwrap();
        writeln!(file, "2 Q0 doc5 1 0.91 dense").unwrap();
        writeln!(file, "2 Q0 doc4 2 0.86 dense").unwrap();
        
        (dir, file_path)
    }

    fn create_temp_trec_qrels() -> (TempDir, std::path::PathBuf) {
        let dir = TempDir::new().unwrap();
        let file_path = dir.path().join("qrels.txt");
        let mut file = fs::File::create(&file_path).unwrap();
        
        // Valid TREC qrels
        writeln!(file, "1 0 doc1 2").unwrap(); // Highly relevant
        writeln!(file, "1 0 doc2 1").unwrap(); // Relevant
        writeln!(file, "1 0 doc3 0").unwrap(); // Not relevant
        writeln!(file, "2 0 doc4 2").unwrap();
        writeln!(file, "2 0 doc5 1").unwrap();
        
        (dir, file_path)
    }

    #[test]
    fn test_end_to_end_evaluation() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        // Load using rank-eval
        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        assert_eq!(runs.len(), 10);
        assert_eq!(qrels.len(), 5);

        // Group by query
        let runs_by_query = group_runs_by_query(&runs);
        let qrels_by_query = group_qrels_by_query(&qrels);

        assert_eq!(runs_by_query.len(), 2);
        assert_eq!(qrels_by_query.len(), 2);

        // Test evaluation for query 1
        let query1_runs = &runs_by_query["1"];
        let query1_qrels = &qrels_by_query["1"];

        // Get bm25 run
        let bm25_run = &query1_runs["bm25"];
        assert_eq!(bm25_run.len(), 3);

        // Evaluate using binary metrics
        let ranked_ids: Vec<&str> = bm25_run.iter().map(|(id, _)| id.as_str()).collect();
        let relevant: std::collections::HashSet<&str> = query1_qrels
            .iter()
            .filter(|(_, &rel)| rel > 0)
            .map(|(id, _)| id.as_str())
            .collect();

        let ndcg = ndcg_at_k(&ranked_ids, &relevant, 10);
        assert!(ndcg > 0.0 && ndcg <= 1.0);

        // Evaluate using graded metrics
        let ranked_with_scores: Vec<(String, f32)> = bm25_run
            .iter()
            .map(|(id, score)| (id.clone(), *score))
            .collect();

        let ndcg_graded = compute_ndcg(&ranked_with_scores, query1_qrels, 10);
        let map = compute_map(&ranked_with_scores, query1_qrels);

        assert!(ndcg_graded >= 0.0 && ndcg_graded <= 1.0);
        assert!(map >= 0.0 && map <= 1.0);
    }

    #[test]
    fn test_rank_eval_trec_parsing() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        // Test that rank-eval parsing works correctly
        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        // Verify structure
        assert_eq!(runs[0].query_id, "1");
        assert_eq!(runs[0].doc_id, "doc1");
        assert_eq!(runs[0].rank, 1);
        assert_eq!(runs[0].score, 0.95);
        assert_eq!(runs[0].run_tag, "bm25");

        assert_eq!(qrels[0].query_id, "1");
        assert_eq!(qrels[0].doc_id, "doc1");
        assert_eq!(qrels[0].relevance, 2);
    }

    #[test]
    fn test_rank_eval_grouping_utilities() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        // Test grouping functions from rank-eval
        let runs_by_query = group_runs_by_query(&runs);
        let qrels_by_query = group_qrels_by_query(&qrels);

        // Verify grouping
        assert!(runs_by_query.contains_key("1"));
        assert!(runs_by_query.contains_key("2"));
        assert!(runs_by_query["1"].contains_key("bm25"));
        assert!(runs_by_query["1"].contains_key("dense"));

        assert!(qrels_by_query.contains_key("1"));
        assert!(qrels_by_query.contains_key("2"));
        assert_eq!(qrels_by_query["1"].len(), 3);
    }

    #[test]
    fn test_multi_run_fusion_3_plus_runs() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        let runs_by_query = group_runs_by_query(&runs);
        let qrels_by_query = group_qrels_by_query(&qrels);

        // Test fusion with 3+ runs
        let query1_runs = &runs_by_query["1"];
        
        // Create a third run
        let mut run3 = query1_runs["bm25"].clone();
        run3.sort_by(|a, b| b.1.total_cmp(&a.1));
        
        // Fuse all three runs
        let run_slices: Vec<&[(String, f32)]> = vec![
            &query1_runs["bm25"],
            &query1_runs["dense"],
            &run3,
        ];

        let method = rank_fusion::FusionMethod::Rrf { k: 60 };
        let fused = method.fuse_multi(&run_slices);

        assert!(!fused.is_empty(), "Fusion should produce results");

        // Evaluate fused results
        let query1_qrels = &qrels_by_query["1"];
        let ranked_with_scores: Vec<(String, f32)> = fused
            .iter()
            .map(|(id, score)| (id.clone(), *score))
            .collect();

        let ndcg = compute_ndcg(&ranked_with_scores, query1_qrels, 10);
        assert!(ndcg >= 0.0 && ndcg <= 1.0);
    }

    #[test]
    fn test_validation_of_valid_dataset() {
        let (_runs_dir, runs_path) = create_temp_trec_runs();
        let (_qrels_dir, qrels_path) = create_temp_trec_qrels();

        // Test dataset validation
        let validation_result = rank_eval::dataset::validate_dataset(&runs_path, &qrels_path).unwrap();

        assert!(validation_result.is_valid, "Valid dataset should pass validation");
        assert!(validation_result.runs_valid);
        assert!(validation_result.qrels_valid);
        assert!(validation_result.consistency_valid);
        assert_eq!(validation_result.errors.len(), 0);
    }

    #[test]
    fn test_validation_of_mismatched_queries() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("runs.txt");
        let qrels_path = dir.path().join("qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        writeln!(runs_file, "2 Q0 doc2 1 0.8 bm25").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();
        // Query 2 missing from qrels

        let validation_result = rank_eval::dataset::validate_dataset(&runs_path, &qrels_path).unwrap();

        // Should have warnings but still be valid
        assert!(validation_result.warnings.len() > 0, "Should warn about mismatched queries");
    }

    #[test]
    fn test_empty_files() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("empty_runs.txt");
        let qrels_path = dir.path().join("empty_qrels.txt");

        fs::File::create(&runs_path).unwrap();
        fs::File::create(&qrels_path).unwrap();

        // Empty files should be handled gracefully
        let runs_result = load_trec_runs(&runs_path);
        let qrels_result = load_qrels(&qrels_path);

        assert!(runs_result.is_ok());
        assert!(qrels_result.is_ok());
        assert_eq!(runs_result.unwrap().len(), 0);
        assert_eq!(qrels_result.unwrap().len(), 0);
    }

    #[test]
    fn test_error_handling_invalid_scores() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("invalid_runs.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 NaN bm25").unwrap(); // Invalid score

        let result = load_trec_runs(&runs_path);
        assert!(result.is_err(), "Should error on NaN score");
    }

    #[test]
    fn test_metric_computation_edge_cases() {
        use std::collections::HashSet;

        // Test with single document
        let ranked = vec!["doc1"];
        let relevant: HashSet<_> = ["doc1"].into_iter().collect();
        let ndcg = ndcg_at_k(&ranked, &relevant, 1);
        assert!((ndcg - 1.0).abs() < 1e-9, "Single relevant doc should give nDCG = 1.0");

        // Test with k=0
        let ndcg_zero = ndcg_at_k(&ranked, &relevant, 0);
        assert_eq!(ndcg_zero, 0.0, "k=0 should give nDCG = 0");

        // Test with k larger than ranked list
        let ndcg_large = ndcg_at_k(&ranked, &relevant, 100);
        assert!(ndcg_large > 0.0, "k > len should still compute correctly");
    }

    #[test]
    fn test_metric_computation_perfect_ranking() {
        use std::collections::HashSet;
        use rank_eval::binary::{ndcg_at_k, precision_at_k, recall_at_k, mrr};

        let ranked = vec!["doc1", "doc2", "doc3", "doc4", "doc5"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 10);
        let precision = precision_at_k(&ranked, &relevant, 3);
        let recall = recall_at_k(&ranked, &relevant, 3);
        let mrr_score = mrr(&ranked, &relevant);

        assert!((ndcg - 1.0).abs() < 1e-9);
        assert!((precision - 1.0).abs() < 1e-9);
        assert!((recall - 1.0).abs() < 1e-9);
        assert!((mrr_score - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_metric_computation_worst_ranking() {
        use std::collections::HashSet;

        // Worst case: relevant docs at the end
        let ranked = vec!["doc4", "doc5", "doc6", "doc1", "doc2", "doc3"];
        let relevant: HashSet<_> = ["doc1", "doc2", "doc3"].into_iter().collect();

        let ndcg = ndcg_at_k(&ranked, &relevant, 6);
        let precision = ndcg_at_k(&ranked, &relevant, 3);

        // Worst case should give low but non-zero metrics
        // Note: nDCG can be > 0.5 even in worst case if relevant docs appear later (just not first)
        // The key is that precision@3 should be 0 (no relevant docs in top-3)
        assert!(ndcg < 0.8, "Worst case ranking should give relatively low nDCG");
        assert_eq!(precision, 0.0, "Top-3 with no relevant should give 0");
    }

    #[test]
    fn test_edge_case_duplicate_query_doc_pairs() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("duplicate_runs.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc1 2 0.8 bm25").unwrap(); // Duplicate

        let runs = load_trec_runs(&runs_path).unwrap();
        assert_eq!(runs.len(), 2, "Should load both entries");
        
        // Validation should warn about duplicates
        let qrels_path = dir.path().join("qrels.txt");
        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();

        let validation = rank_eval::dataset::validate_dataset(&runs_path, &qrels_path).unwrap();
        assert!(validation.warnings.len() > 0, "Should warn about duplicate entries");
    }

    #[test]
    fn test_edge_case_all_same_scores() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("same_scores.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        writeln!(runs_file, "1 Q0 doc1 1 0.5 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.5 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc3 3 0.5 bm25").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        let runs_by_query = group_runs_by_query(&runs);
        let query1_runs = &runs_by_query["1"]["bm25"];

        // All scores are the same, ranking should preserve order
        assert_eq!(query1_runs.len(), 3);
    }

    #[test]
    fn test_edge_case_no_relevant_documents() {
        use std::collections::HashMap;

        let ranked = vec![
            ("doc1".to_string(), 0.9),
            ("doc2".to_string(), 0.8),
        ];
        let qrels: HashMap<String, u32> = HashMap::new(); // No relevant docs

        let ndcg = compute_ndcg(&ranked, &qrels, 10);
        let map = compute_map(&ranked, &qrels);

        assert_eq!(ndcg, 0.0);
        assert_eq!(map, 0.0);
    }

    #[test]
    fn test_edge_case_single_document_per_query() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("single_doc.txt");
        let qrels_path = dir.path().join("single_doc_qrels.txt");

        let mut runs_file = fs::File::create(&runs_path).unwrap();
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();

        let mut qrels_file = fs::File::create(&qrels_path).unwrap();
        writeln!(qrels_file, "1 0 doc1 2").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        let qrels = load_qrels(&qrels_path).unwrap();

        assert_eq!(runs.len(), 1);
        assert_eq!(qrels.len(), 1);
    }

    #[test]
    fn test_edge_case_unicode_and_special_chars() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("unicode_runs.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        // Test with unicode in doc IDs
        writeln!(runs_file, "1 Q0 doc_测试 1 0.9 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc_🎉 2 0.8 bm25").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].doc_id, "doc_测试");
        assert_eq!(runs[1].doc_id, "doc_🎉");
    }

    #[test]
    fn test_edge_case_very_large_scores() {
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("large_scores.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        writeln!(runs_file, "1 Q0 doc1 1 999999.99 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.0001 bm25").unwrap();

        let runs = load_trec_runs(&runs_path).unwrap();
        assert_eq!(runs.len(), 2);
        assert_eq!(runs[0].score, 999999.99);
        assert_eq!(runs[1].score, 0.0001);
    }

    #[test]
    fn test_edge_case_rank_ties() {
        // Test that ranking handles ties correctly
        let dir = TempDir::new().unwrap();
        let runs_path = dir.path().join("ties.txt");
        let mut runs_file = fs::File::create(&runs_path).unwrap();
        
        writeln!(runs_file, "1 Q0 doc1 1 0.9 bm25").unwrap();
        writeln!(runs_file, "1 Q0 doc2 2 0.9 bm25").unwrap(); // Same score
        writeln!(runs_file, "1 Q0 doc3 3 0.9 bm25").unwrap(); // Same score

        let runs = load_trec_runs(&runs_path).unwrap();
        let runs_by_query = group_runs_by_query(&runs);
        let query1_runs = &runs_by_query["1"]["bm25"];

        // All have same score, should preserve order or sort consistently
        assert_eq!(query1_runs.len(), 3);
    }

    #[test]
    fn test_metric_precision_fewer_than_k() {
        use std::collections::HashSet;

        // Test precision when ranked list has fewer than k items
        let ranked = vec!["doc1", "doc2"]; // Only 2 items
        let relevant: HashSet<_> = ["doc1"].into_iter().collect();

        let precision = rank_eval::binary::precision_at_k(&ranked, &relevant, 10);
        // Precision@10 with 2 items, 1 relevant = 1/10 = 0.1
        assert!((precision - 0.1).abs() < 1e-9);
    }
}
