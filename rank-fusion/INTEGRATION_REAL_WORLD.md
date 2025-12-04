# Real-World Integration Guide

This guide provides **actual, working** integration examples with popular search and vector databases. Unlike simulated examples, these use real client libraries and can be run against actual services.

## Prerequisites

1. **Elasticsearch/OpenSearch**: Running on localhost:9200 (or configure URL)
2. **Qdrant/Pinecone/Weaviate**: Running vector database
3. **Optional dependencies**: Enable `integration-examples` feature

## Setup

### 1. Add Optional Dependencies

Add to `Cargo.toml`:

```toml
[features]
integration-examples = ["dep:elasticsearch", "dep:qdrant-client", "dep:reqwest"]

[dependencies]
# ... existing dependencies ...

# Optional integration dependencies
elasticsearch = { version = "8.5", optional = true }
qdrant-client = { version = "1.7", optional = true }
reqwest = { version = "0.11", features = ["json"], optional = true }
tokio = { version = "1.0", features = ["full"], optional = true }
```

### 2. Enable Feature

```bash
cargo run --example real_world_elasticsearch_actual --features integration-examples
```

## Integration Examples

### Elasticsearch + Qdrant Hybrid Search

**File**: `examples/real_world_elasticsearch_actual.rs`

**What it does**:
- Searches Elasticsearch using BM25 (keyword matching)
- Searches Qdrant using dense vector similarity
- Fuses results with RRF
- Validates and returns top-k results

**Usage**:
```bash
# Start Elasticsearch
docker run -d -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8.5.0

# Start Qdrant
docker run -d -p 6333:6333 qdrant/qdrant

# Run example
cargo run --example real_world_elasticsearch_actual --features integration-examples
```

**Code Structure**:
```rust
// 1. Search Elasticsearch
let bm25_results = search_elasticsearch(query, 50)?;

// 2. Search Qdrant
let query_embedding = get_query_embedding(query)?;
let dense_results = search_qdrant(&query_embedding, 50)?;

// 3. Fuse with RRF
let fused = rrf_multi(&[&bm25_results[..], &dense_results[..]], config)?;

// 4. Validate
let validation = validate(&fused, false, Some(20))?;
```

### Meilisearch + Pinecone

**File**: `examples/real_world_meilisearch_pinecone.rs` (to be created)

Similar pattern but using Meilisearch (BM25) and Pinecone (vector DB).

### LangChain Integration (Python)

**File**: `rank-fusion-python/examples/langchain_integration.py`

**What it does**:
- Uses LangChain retrievers (BM25Retriever, VectorStoreRetriever)
- Extracts results and converts to rank-fusion format
- Fuses results
- Returns to LangChain format

**Usage**:
```python
from langchain.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
import rank_fusion

# Setup retrievers
bm25_retriever = BM25Retriever.from_documents(documents)
vector_store = FAISS.from_documents(documents, embeddings)
vector_retriever = vector_store.as_retriever()

# Retrieve
bm25_docs = bm25_retriever.get_relevant_documents(query)
vector_docs = vector_retriever.get_relevant_documents(query)

# Convert to rank-fusion format
def extract_ranked_list(docs):
    return [(doc.metadata.get("id", str(i)), getattr(doc, "score", 1.0/(i+1))) 
            for i, doc in enumerate(docs)]

bm25_ranked = extract_ranked_list(bm25_docs)
vector_ranked = extract_ranked_list(vector_docs)

# Fuse
fused = rank_fusion.rrf(bm25_ranked, vector_ranked, k=60)

# Get top documents
top_ids = [id for id, _ in fused[:10]]
top_docs = [get_doc_by_id(id) for id in top_ids]
```

### LlamaIndex Integration (Python)

**File**: `rank-fusion-python/examples/llamaindex_integration.py`

Similar pattern for LlamaIndex query engines.

## Error Handling in Production

All integration examples should handle:

1. **Connection errors**: Retry with exponential backoff
2. **Timeout errors**: Fallback to single retriever
3. **Empty results**: Return empty list, don't panic
4. **Validation failures**: Log warnings, continue with results

Example:

```rust
fn hybrid_search_with_fallback(query: &str) -> Vec<(String, f32)> {
    // Try both retrievers
    let bm25_results = search_elasticsearch(query, 50)
        .unwrap_or_else(|e| {
            eprintln!("Elasticsearch error: {}, using empty results", e);
            Vec::new()
        });
    
    let dense_results = search_qdrant(query, 50)
        .unwrap_or_else(|e| {
            eprintln!("Qdrant error: {}, using empty results", e);
            Vec::new()
        });
    
    // If both fail, return empty
    if bm25_results.is_empty() && dense_results.is_empty() {
        return Vec::new();
    }
    
    // If one fails, return the other
    if bm25_results.is_empty() {
        return dense_results;
    }
    if dense_results.is_empty() {
        return bm25_results;
    }
    
    // Both succeeded, fuse
    rrf(&bm25_results, &dense_results)
}
```

## Testing Integrations

### Local Testing

1. Use Docker Compose to start services:
```yaml
# docker-compose.yml
version: '3.8'
services:
  elasticsearch:
    image: elasticsearch:8.5.0
    ports:
      - "9200:9200"
    environment:
      - discovery.type=single-node
      - xpack.security.enabled=false
  
  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
```

2. Run integration tests:
```bash
cargo test --features integration-examples --test integration_real_world
```

### CI/CD Testing

Integration examples should be feature-gated and skipped in CI unless explicitly enabled:

```rust
#[cfg(test)]
mod tests {
    #[test]
    #[cfg(feature = "integration-examples")]
    fn test_elasticsearch_integration() {
        // Only runs with --features integration-examples
    }
}
```

## Performance Considerations

1. **Parallel retrieval**: Use `tokio::join!` or `futures::join!` to search both retrievers concurrently
2. **Caching**: Cache query embeddings and frequent queries
3. **Timeout**: Set reasonable timeouts (e.g., 2s per retriever)
4. **Connection pooling**: Reuse client connections

Example:

```rust
use tokio::time::{timeout, Duration};

async fn parallel_hybrid_search(query: &str) -> Result<Vec<(String, f32)>, Box<dyn std::error::Error>> {
    let query_embedding = get_query_embedding(query).await?;
    
    // Search both in parallel with timeout
    let (bm25_result, dense_result) = tokio::join!(
        timeout(Duration::from_secs(2), search_elasticsearch_async(query, 50)),
        timeout(Duration::from_secs(2), search_qdrant_async(&query_embedding, 50)),
    );
    
    let bm25 = bm25_result??.unwrap_or_default();
    let dense = dense_result??.unwrap_or_default();
    
    Ok(rrf(&bm25, &dense))
}
```

## Next Steps

1. **Complete implementations**: Replace template code with actual client calls
2. **Add more integrations**: Meilisearch, Pinecone, Weaviate, Milvus
3. **Add monitoring**: Logging, metrics, tracing
4. **Add benchmarks**: Measure latency, throughput

