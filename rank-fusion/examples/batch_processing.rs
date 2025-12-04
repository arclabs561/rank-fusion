//! Batch processing example with parallelization hints.
//!
//! This example demonstrates how to process multiple queries in parallel
//! using `rank-fusion` for high-throughput scenarios.
//!
//! Run: `cargo run --example batch_processing`
//!
//! For actual parallelization, use `rayon`:
//! ```toml
//! [dependencies]
//! rayon = "1.8"
//! ```

use rank_fusion::{rrf_multi, RrfConfig, validate::validate};
use std::time::Instant;

/// Process a single query through the fusion pipeline.
fn process_query(
    query_id: &str,
    bm25_results: &[(String, f32)],
    dense_results: &[(String, f32)],
) -> Vec<(String, f32)> {
    let fused = rrf_multi(
        &[bm25_results, dense_results],
        RrfConfig::new(60).with_top_k(10),
    );

    // Validate results
    let validation = validate(&fused, false, Some(10));
    if !validation.is_valid {
        eprintln!("Query {}: Validation errors: {:?}", query_id, validation.errors);
    }

    fused
}

/// Process queries sequentially (baseline).
fn process_sequential(queries: &[(String, Vec<(String, f32)>, Vec<(String, f32)>)]) -> Vec<(String, Vec<(String, f32)>)> {
    let start = Instant::now();
    let mut results = Vec::with_capacity(queries.len());

    for (query_id, bm25, dense) in queries {
        let fused = process_query(query_id, bm25, dense);
        results.push((query_id.clone(), fused));
    }

    let elapsed = start.elapsed();
    println!("Sequential processing: {} queries in {:?} ({:.2} queries/sec)",
        queries.len(), elapsed, queries.len() as f64 / elapsed.as_secs_f64());
    
    results
}

/// Process queries in parallel using rayon (if available).
///
/// To use this, add `rayon = "1.8"` to your `Cargo.toml`.
/// Note: This example shows the pattern but rayon is not a dependency of this crate.
/// You would need to add rayon as a dependency in your own project.
///
/// Example usage in your project:
/// ```rust
/// use rayon::prelude::*;
///
/// let results: Vec<_> = queries
///     .par_iter()
///     .map(|(query_id, bm25, dense)| {
///         let fused = process_query(query_id, bm25, dense);
///         (query_id.clone(), fused)
///     })
///     .collect();
/// ```
fn process_parallel_rayon(queries: &[(String, Vec<(String, f32)>, Vec<(String, f32)>)]) -> Vec<(String, Vec<(String, f32)>)> {
    // For this example, we'll use sequential processing as a fallback
    // In your project, add rayon and use par_iter() for parallel processing
    println!("   (Using sequential processing as fallback - add rayon for parallel)");
    process_sequential(queries)
}

/// Process queries in batches (chunked processing).
///
/// Useful when you have many queries but want to limit memory usage.
fn process_batched(
    queries: &[(String, Vec<(String, f32)>, Vec<(String, f32)>)],
    batch_size: usize,
) -> Vec<(String, Vec<(String, f32)>)> {
    let start = Instant::now();
    let mut results = Vec::with_capacity(queries.len());

    for chunk in queries.chunks(batch_size) {
        for (query_id, bm25, dense) in chunk {
            let fused = process_query(query_id, bm25, dense);
            results.push((query_id.clone(), fused));
        }
    }

    let elapsed = start.elapsed();
    println!("Batched processing (batch_size={}): {} queries in {:?} ({:.2} queries/sec)",
        batch_size, queries.len(), elapsed, queries.len() as f64 / elapsed.as_secs_f64());
    
    results
}

fn main() {
    println!("=== Batch Processing Example ===\n");

    // Generate mock queries with BM25 and dense results
    let queries: Vec<(String, Vec<(String, f32)>, Vec<(String, f32)>)> = (0..100)
        .map(|i| {
            let query_id = format!("query_{}", i);
            
            // Mock BM25 results
            let bm25: Vec<(String, f32)> = (0..20)
                .map(|j| (format!("doc_{}_{}", i, j), 10.0 - j as f32 * 0.5))
                .collect();
            
            // Mock dense results (different documents)
            let dense: Vec<(String, f32)> = (0..20)
                .map(|j| (format!("doc_{}_{}", i, j + 10), 0.9 - j as f32 * 0.05))
                .collect();
            
            (query_id, bm25, dense)
        })
        .collect();

    println!("Generated {} queries for processing\n", queries.len());

    // Sequential processing (baseline)
    println!("1. Sequential Processing:");
    let _results_seq = process_sequential(&queries);

    // Batched processing
    println!("\n2. Batched Processing:");
    let _results_batched = process_batched(&queries, 10);

    // Parallel processing example (shows pattern, requires rayon in your project)
    println!("\n3. Parallel Processing (rayon):");
    println!("   Note: This example shows the pattern. To use rayon:");
    println!("   1. Add 'rayon = \"1.8\"' to your Cargo.toml");
    println!("   2. Use rayon's par_iter() for parallel processing");
    println!("   3. See process_parallel_rayon() for example code");
    let _results_par = process_parallel_rayon(&queries);

    println!("\n💡 Performance Tips:");
    println!("  - For <100 queries: sequential is usually fine");
    println!("  - For 100-1000 queries: batching helps with memory");
    println!("  - For >1000 queries: use rayon for parallelization");
    println!("  - Fusion itself is fast (~13μs for 100 items), so I/O is usually the bottleneck");
    println!("  - Consider async/await for I/O-bound workloads (retriever calls)");
}

