//! Real-world integration example: Elasticsearch + Vector DB hybrid search.
//!
//! This example demonstrates actual integration with Elasticsearch and Qdrant
//! using their official Rust clients.
//!
//! **Prerequisites:**
//! None! This example uses mock data by default and can run without Elasticsearch or Qdrant.
//!
//! **To use with real services:**
//! 1. Elasticsearch running on localhost:9200
//! 2. Qdrant running on localhost:6333
//! 3. Replace the mock implementations in `search_elasticsearch()` and `search_qdrant()`
//!    with real client code (see comments in those functions)
//! 4. Add dependencies to Cargo.toml:
//!    ```toml
//!    [dependencies]
//!    elasticsearch = { version = "...", optional = true }
//!    qdrant-client = { version = "...", optional = true }
//!    ```
//!
//! **Note**: The example currently uses mock data to demonstrate the fusion pipeline
//! without requiring external services. Replace the mock implementations with real
//! client code when ready.

fn main() {
    use rank_fusion::{rrf_multi, RrfConfig, validate::validate};
    
    println!("=== Real-World Hybrid Search Integration (Mock Mode) ===\n");
    println!("Note: Using mock data. Replace mock functions with real client code for production.\n");
    
    let query = "machine learning";
    
    // Step 1: Search Elasticsearch (BM25) - mock
    println!("Step 1: Searching Elasticsearch (BM25)...");
    let bm25_results = search_elasticsearch(query, 50);
    println!("  Found {} results from Elasticsearch", bm25_results.len());
    
    // Step 2: Search Qdrant (Dense vectors) - mock
    println!("\nStep 2: Searching Qdrant (Dense vectors)...");
    let query_embedding = get_query_embedding(query);
    let dense_results = search_qdrant(&query_embedding, 50);
    println!("  Found {} results from Qdrant", dense_results.len());
    
    // Step 3: Fuse results with RRF
    println!("\nStep 3: Fusing results with RRF...");
    let lists = vec![&bm25_results[..], &dense_results[..]];
    let config = RrfConfig::default().with_top_k(20);
    let fused = rrf_multi(&lists, config);
    
    // Step 4: Validate results
    let validation = validate(&fused, false, Some(20));
    if !validation.is_valid {
        eprintln!("Warning: Validation errors: {:?}", validation.errors);
    }
    
    println!("\nStep 4: Top 10 Fused Results:");
    for (i, (doc_id, score)) in fused.iter().take(10).enumerate() {
        println!("  {}. {} (score: {:.6})", i + 1, doc_id, score);
    }
    
    println!("\n✅ Hybrid search complete!");
}

fn search_elasticsearch(query: &str, top_k: usize) -> Vec<(String, f32)> {
    // Mock implementation for demonstration
    // In production, replace with actual Elasticsearch client:
    //
    // ```rust
    // use elasticsearch::{Elasticsearch, SearchParts};
    // use elasticsearch::http::transport::Transport;
    // use serde_json::{json, Value};
    //
    // let client = Elasticsearch::new(Transport::single_node("http://localhost:9200")?)?;
    // let response = client
    //     .search(SearchParts::Index(&["documents"]))
    //     .body(json!({
    //         "query": {
    //             "match": {
    //                 "text": query
    //             }
    //         },
    //         "size": top_k
    //     }))
    //     .send()
    //     .await?;
    //
    // let body: Value = response.json().await?;
    // let hits = body["hits"]["hits"].as_array().unwrap_or(&vec![]);
    // let results: Vec<(String, f32)> = hits
    //     .iter()
    //     .map(|hit| {
    //         let id = hit["_id"].as_str().unwrap_or("").to_string();
    //         let score = hit["_score"].as_f64().unwrap_or(0.0) as f32;
    //         (id, score)
    //     })
    //     .collect();
    // Ok(results)
    // ```
    
    // Mock results for demonstration (simulating BM25 scores)
    println!("  [Mock] Searching Elasticsearch for: \"{}\"", query);
    let mock_results: Vec<(String, f32)> = (0..top_k)
        .map(|i| {
            let doc_id = format!("es_doc_{}", i + 1);
            // Simulate BM25 scores (typically 10-20 range)
            let score = 15.0 - (i as f32 * 0.3);
            (doc_id, score)
        })
        .collect();
    
    mock_results
}

fn search_qdrant(query_embedding: &[f32], top_k: usize) -> Vec<(String, f32)> {
    // Mock implementation for demonstration
    // In production, replace with actual Qdrant client:
    //
    // ```rust
    // use qdrant_client::prelude::*;
    // use qdrant_client::qdrant::{SearchPoints, SearchParams};
    //
    // let config = QdrantClientConfig::from_url("http://localhost:6333");
    // let client = QdrantClient::new(Some(config))?;
    //
    // let search_result = client
    //     .search_points(&SearchPoints {
    //         collection_name: "documents".to_string(),
    //         vector: query_embedding.to_vec(),
    //         limit: top_k as u64,
    //         with_payload: Some(true.into()),
    //         ..Default::default()
    //     })
    //     .await?;
    //
    // let results: Vec<(String, f32)> = search_result.result
    //     .iter()
    //     .map(|point| {
    //         let id = point.id.as_ref().and_then(|id| match id {
    //             PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Uuid(uuid)) } => Some(uuid.clone()),
    //             PointId { point_id_options: Some(qdrant_client::qdrant::point_id::PointIdOptions::Num(num)) } => Some(num.to_string()),
    //             _ => None,
    //         }).unwrap_or_else(|| "unknown".to_string());
    //         let score = point.score;
    //         (id, score as f32)
    //     })
    //     .collect();
    // Ok(results)
    // ```
    
    // Mock results for demonstration (simulating cosine similarity scores)
    println!("  [Mock] Searching Qdrant with embedding (dim={})", query_embedding.len());
    let mock_results: Vec<(String, f32)> = (0..top_k)
        .map(|i| {
            let doc_id = format!("qdrant_doc_{}", i + 1);
            // Simulate cosine similarity scores (typically 0.7-0.95 range)
            let score = 0.95 - (i as f32 * 0.02);
            (doc_id, score)
        })
        .collect();
    
    mock_results
}

fn get_query_embedding(_query: &str) -> Vec<f32> {
    // In production, this would call an embedding model (e.g., via ONNX, HTTP API, etc.)
    // For now, return a mock embedding
    vec![0.1; 384] // Example: 384-dimensional embedding
}

