//! Real-world integration example: Elasticsearch + Vector DB hybrid search.
//!
//! This example demonstrates how to integrate `rank-fusion` with Elasticsearch
//! (BM25) and a vector database (dense retrieval) for production RAG pipelines.
//!
//! Run: `cargo run --example real_world_elasticsearch --features wasm`
//!
//! Note: This example uses mock clients. In production, replace with actual
//! Elasticsearch and vector DB clients.

use rank_fusion::{rrf_multi, RrfConfig, validate::validate};

/// Mock Elasticsearch client (replace with actual client in production).
struct ElasticsearchClient;

impl ElasticsearchClient {
    fn new(_url: &str) -> Self {
        Self
    }

    /// Search using BM25 (keyword matching).
    fn search(&self, query: &str, top_k: usize) -> Vec<(String, f32)> {
        // In production, this would call Elasticsearch's search API
        // Example: POST /_search with {"query": {"match": {"text": query}}}
        println!("[Elasticsearch] Searching for: \"{}\" (top_k={})", query, top_k);
        
        // Mock results (simulating BM25 scores)
        vec![
            ("doc_1".to_string(), 12.5),
            ("doc_3".to_string(), 11.2),
            ("doc_5".to_string(), 9.8),
            ("doc_2".to_string(), 8.5),
        ]
        .into_iter()
        .take(top_k)
        .collect()
    }
}

/// Mock vector database client (replace with actual client in production).
struct VectorDBClient;

impl VectorDBClient {
    fn new(_url: &str) -> Self {
        Self
    }

    /// Search using dense vector similarity (semantic search).
    fn search(&self, query_embedding: &[f32], top_k: usize) -> Vec<(String, f32)> {
        // In production, this would call vector DB's similarity search
        // Example: Qdrant, Pinecone, Weaviate, etc.
        println!("[VectorDB] Searching with embedding (dim={}, top_k={})", query_embedding.len(), top_k);
        
        // Mock results (simulating cosine similarity scores)
        vec![
            ("doc_2".to_string(), 0.95),
            ("doc_4".to_string(), 0.89),
            ("doc_1".to_string(), 0.85),
            ("doc_6".to_string(), 0.82),
        ]
        .into_iter()
        .take(top_k)
        .collect()
    }
}

/// Hybrid search pipeline combining Elasticsearch and vector DB.
struct HybridSearchPipeline {
    es_client: ElasticsearchClient,
    vector_client: VectorDBClient,
}

impl HybridSearchPipeline {
    fn new(es_url: &str, vector_url: &str) -> Self {
        Self {
            es_client: ElasticsearchClient::new(es_url),
            vector_client: VectorDBClient::new(vector_url),
        }
    }

    /// Perform hybrid search: BM25 + dense retrieval → RRF fusion.
    ///
    /// # Arguments
    /// * `query` - User query string
    /// * `query_embedding` - Dense embedding of the query (for vector search)
    /// * `top_k_per_retriever` - Number of results to retrieve from each retriever
    /// * `final_top_k` - Number of final fused results to return
    ///
    /// # Returns
    /// Fused results sorted by RRF score (descending)
    pub fn search(
        &self,
        query: &str,
        query_embedding: &[f32],
        top_k_per_retriever: usize,
        final_top_k: usize,
    ) -> Vec<(String, f32)> {
        println!("\n=== Hybrid Search Pipeline ===\n");

        // Step 1: Retrieve from Elasticsearch (BM25)
        let bm25_results = self.es_client.search(query, top_k_per_retriever);
        println!("BM25 results: {:?}\n", bm25_results);

        // Step 2: Retrieve from vector DB (dense)
        let dense_results = self.vector_client.search(query_embedding, top_k_per_retriever);
        println!("Dense results: {:?}\n", dense_results);

        // Step 3: Fuse using RRF (handles incompatible score scales)
        let fused = rrf_multi(
            &[bm25_results.as_slice(), dense_results.as_slice()],
            RrfConfig::new(60).with_top_k(final_top_k),
        );
        println!("Fused results (RRF, top {}):", final_top_k);
        for (i, (id, score)) in fused.iter().enumerate() {
            println!("  {}. {}: {:.4}", i + 1, id, score);
        }

        // Step 4: Validate results
        let validation = validate(&fused, false, Some(final_top_k));
        if !validation.is_valid {
            eprintln!("⚠️  Validation errors: {:?}", validation.errors);
        }
        if !validation.warnings.is_empty() {
            println!("ℹ️  Validation warnings: {:?}", validation.warnings);
        } else {
            println!("✅ Results validated successfully");
        }

        fused
    }
}

fn main() {
    // Initialize clients (in production, use actual connection strings)
    let pipeline = HybridSearchPipeline::new(
        "http://localhost:9200",
        "http://localhost:6333", // Example: Qdrant
    );

    // User query
    let query = "How does Rust prevent memory leaks?";
    
    // In production, generate embedding using a model (e.g., sentence-transformers)
    // For this example, use a mock embedding
    let query_embedding = vec![0.1; 384]; // Mock 384-dimensional embedding

    // Perform hybrid search
    let results = pipeline.search(query, &query_embedding, 10, 5);

    println!("\n=== Final Results ===");
    for (i, (doc_id, score)) in results.iter().enumerate() {
        println!("{}. {} (score: {:.4})", i + 1, doc_id, score);
    }

    // In production, you would:
    // 1. Fetch full document content for top results
    // 2. Rerank with a cross-encoder (optional, using rank-refine)
    // 3. Send to LLM for generation
    println!("\n💡 Next steps in production:");
    println!("  1. Fetch document content for top results");
    println!("  2. Optionally rerank with cross-encoder (rank-refine)");
    println!("  3. Send context to LLM for answer generation");
}

