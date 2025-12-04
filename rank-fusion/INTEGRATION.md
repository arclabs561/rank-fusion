# Integration Guides

This document provides integration examples for common RAG/search stacks.

## Python Integration

### Using with LangChain

**Note**: LangChain APIs may vary by version. This example shows the general pattern.

```python
import rank_fusion

# Option 1: Using LangChain retrievers (version-dependent API)
# from langchain.retrievers import BM25Retriever
# from langchain.vectorstores import VectorStore

# Get results from multiple retrievers
# bm25_retriever = BM25Retriever.from_documents(documents)
# vector_store = VectorStore.from_documents(documents, embeddings)
# 
# bm25_results = bm25_retriever.get_relevant_documents(query)
# vector_results = vector_store.as_retriever().get_relevant_documents(query)

# Option 2: Generic pattern (works across LangChain versions)
# Extract document IDs and scores from LangChain results
def extract_ranked_list(langchain_results):
    """Convert LangChain results to rank-fusion format."""
    ranked = []
    for i, doc in enumerate(langchain_results):
        doc_id = doc.metadata.get("id", doc.metadata.get("source", str(i)))
        # LangChain may not have scores - use rank position as proxy
        score = getattr(doc, "score", 1.0 / (i + 1))
        ranked.append((doc_id, score))
    return ranked

# Example usage (replace with actual LangChain calls):
# bm25_results = bm25_retriever.get_relevant_documents(query)
# vector_results = vector_retriever.get_relevant_documents(query)
# 
# bm25_ranked = extract_ranked_list(bm25_results)
# vector_ranked = extract_ranked_list(vector_results)

# Fuse results
# fused = rank_fusion.rrf(bm25_ranked, vector_ranked, k=60)

# Convert back to LangChain format
# fused_docs = [get_document_by_id(id) for id, _ in fused[:10]]  # Top 10
```

### Using with LlamaIndex

**Note**: LlamaIndex APIs vary by version. This example shows the general pattern.

```python
import rank_fusion

# Option 1: LlamaIndex v0.9+ (query engine API)
# from llama_index.core import VectorStoreIndex, KeywordTableIndex
# 
# vector_index = VectorStoreIndex.from_documents(documents)
# keyword_index = KeywordTableIndex.from_documents(documents)
# 
# vector_query_engine = vector_index.as_query_engine(similarity_top_k=50)
# keyword_query_engine = keyword_index.as_query_engine(similarity_top_k=50)
# 
# vector_response = vector_query_engine.query(query)
# keyword_response = keyword_query_engine.query(query)
# 
# # Extract results
# vector_ranked = [(node.node_id, node.score) for node in vector_response.source_nodes]
# keyword_ranked = [(node.node_id, node.score) for node in keyword_response.source_nodes]

# Option 2: Generic pattern (works across LlamaIndex versions)
def extract_llamaindex_results(response_or_nodes):
    """Convert LlamaIndex results to rank-fusion format."""
    ranked = []
    # Handle both query response and direct node lists
    nodes = getattr(response_or_nodes, "source_nodes", response_or_nodes)
    for i, node in enumerate(nodes):
        node_id = getattr(node, "node_id", getattr(node, "id", str(i)))
        score = getattr(node, "score", 1.0 / (i + 1))
        ranked.append((node_id, score))
    return ranked

# Example usage:
# vector_ranked = extract_llamaindex_results(vector_response)
# keyword_ranked = extract_llamaindex_results(keyword_response)

# Fuse results
# fused = rank_fusion.rrf_multi([vector_ranked, keyword_ranked], k=60)

# Get top-k fused documents
# top_k_ids = [id for id, _ in fused[:10]]
# fused_nodes = [get_node_by_id(id) for id in top_k_ids]
```

### Using with Elasticsearch/OpenSearch

```python
from opensearchpy import OpenSearch
import rank_fusion

client = OpenSearch([...])

# BM25 search
bm25_response = client.search(
    index="documents",
    body={"query": {"match": {"text": query}}}
)
bm25_results = [
    (hit["_id"], hit["_score"])
    for hit in bm25_response["hits"]["hits"]
]

# Vector search
vector_response = client.search(
    index="documents",
    body={
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": 50
                }
            }
        }
    }
)
vector_results = [
    (hit["_id"], hit["_score"])
    for hit in vector_response["hits"]["hits"]
]

# Fuse results (OpenSearch has built-in RRF, but you can use this for custom logic)
fused = rank_fusion.rrf(bm25_results, vector_results, k=60)
```

## Rust Integration

### Using with qdrant

```rust
use qdrant_client::QdrantClient;
use rank_fusion::rrf;

// BM25 search (using tantivy or similar)
let bm25_results = bm25_search(&query)?;

// Vector search with qdrant
let vector_results = client
    .search_points(&SearchPoints {
        collection_name: "documents".to_string(),
        vector: query_embedding,
        limit: 50,
        ..Default::default()
    })
    .await?;

// Convert qdrant results to rank-fusion format
let vector_ranked: Vec<(String, f32)> = vector_results
    .result
    .iter()
    .map(|p| (p.id.to_string(), p.score))
    .collect();

// Fuse results
let fused = rrf(&bm25_results, &vector_ranked);
```

### Using with weaviate

```rust
use weaviate_community::WeaviateClient;
use rank_fusion::rrf;

// Get results from multiple retrievers
let bm25_results = bm25_search(&query)?;
let vector_results = weaviate_client
    .query()
    .get()
    .with_near_vector(query_embedding)
    .with_limit(50)
    .build()
    .execute()
    .await?;

// Convert and fuse
let vector_ranked: Vec<(String, f32)> = vector_results
    .iter()
    .map(|r| (r.id.clone(), r.certainty.unwrap_or(0.0)))
    .collect();

let fused = rrf(&bm25_results, &vector_ranked);
```

## JavaScript/TypeScript Integration

### Using with WebAssembly

See `examples/webassembly.rs` for build instructions and JavaScript API design.

### Using via REST API

Create a Rust service using actix-web or axum:

```rust
use actix_web::{web, App, HttpServer, Result};
use rank_fusion::rrf;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct FusionRequest {
    lists: Vec<Vec<(String, f32)>>,
    k: Option<u32>,
}

#[derive(Serialize)]
struct FusionResponse {
    results: Vec<(String, f32)>,
}

async fn fuse(req: web::Json<FusionRequest>) -> Result<web::Json<FusionResponse>> {
    use rank_fusion::{rrf_multi, RrfConfig};
    
    let slices: Vec<&[(String, f32)]> = req.lists.iter()
        .map(|v| v.as_slice())
        .collect();
    let k = req.k.unwrap_or(60);
    let config = RrfConfig::new(k);
    let fused = rrf_multi(&slices, config);
    Ok(web::Json(FusionResponse { results: fused }))
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new()
            .route("/fuse", web::post().to(fuse))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

Then call from JavaScript:

```javascript
const response = await fetch('http://localhost:8080/fuse', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
        lists: [
            [["d1", 12.5], ["d2", 11.0]],
            [["d2", 0.9], ["d3", 0.8]]
        ],
        k: 60
    })
});
const { results } = await response.json();
```

## C/C++ Integration

### Using C FFI

Create a C-compatible API:

```rust
#[no_mangle]
pub extern "C" fn rrf_fuse(
    list_a: *const RankedItem,
    len_a: usize,
    list_b: *const RankedItem,
    len_b: usize,
    k: u32,
    output: *mut RankedItem,
) -> usize {
    // Convert C arrays to Rust slices
    // Call rrf()
    // Write results to output
    // Return count
}
```

Then use from C:

```c
#include "rank_fusion.h"

RankedItem a[] = {{"d1", 12.5}, {"d2", 11.0}};
RankedItem b[] = {{"d2", 0.9}, {"d3", 0.8}};
RankedItem output[10];

size_t count = rrf_fuse(a, 2, b, 2, 60, output, 10);
```

