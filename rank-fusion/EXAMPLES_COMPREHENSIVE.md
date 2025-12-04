# Comprehensive Examples

Complete, runnable examples for all algorithms and use cases.

## Table of Contents

1. [Basic Fusion Examples](#basic-fusion-examples)
2. [Error Handling Examples](#error-handling-examples)
3. [Real-World Integration Examples](#real-world-integration-examples)
4. [Explainability Examples](#explainability-examples)
5. [Validation Examples](#validation-examples)
6. [Advanced Configuration Examples](#advanced-configuration-examples)

---

## Basic Fusion Examples

### RRF (Reciprocal Rank Fusion)

```rust
use rank_fusion::rrf;

// BM25 results (scores 0-100)
let bm25 = vec![
    ("doc1", 87.5),
    ("doc2", 82.3),
    ("doc3", 78.1),
];

// Dense embedding results (cosine similarity 0-1)
let dense = vec![
    ("doc2", 0.92),
    ("doc1", 0.88),
    ("doc4", 0.85),
];

// RRF finds consensus: doc2 appears high in both lists
let fused = rrf(&bm25, &dense);
// Result: [("doc2", 0.033), ("doc1", 0.032), ("doc3", 0.017), ("doc4", 0.016)]
// doc2 wins (rank 1 in BM25, rank 0 in dense)
```

### RRF with Custom k

```rust
use rank_fusion::{rrf_with_config, RrfConfig};

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];

// k=20: stronger consensus required (top positions dominate more)
let fused = rrf_with_config(&bm25, &dense, RrfConfig::new(20).with_top_k(10));
```

### CombSUM (Score-Based Fusion)

```rust
use rank_fusion::combsum;

// Both lists use same embedding model (compatible scales)
let list1 = vec![("doc1", 0.95), ("doc2", 0.85)];
let list2 = vec![("doc2", 0.90), ("doc3", 0.80)];

// Simple sum of scores
let fused = combsum(&list1, &list2);
// Result: [("doc2", 1.75), ("doc1", 0.95), ("doc3", 0.80)]
```

### CombMNZ (Reward Overlap)

```rust
use rank_fusion::combmnz;

let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];

// Multiplies sum by number of lists containing each doc
let fused = combmnz(&list1, &list2);
// Result: [("doc2", 2.6), ("doc1", 0.9), ("doc3", 0.3)]
// doc2 gets 2× multiplier (appears in both lists)
```

### Weighted Fusion

```rust
use rank_fusion::{weighted, WeightedConfig};

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];

// Trust dense embeddings more (70% weight)
let config = WeightedConfig::new(0.3, 0.7).with_normalize(true);
let fused = weighted(&bm25, &dense, config);
```

### DBSF (Distribution-Based Score Fusion)

```rust
use rank_fusion::dbsf;

// Different score distributions
let list1 = vec![("doc1", 12.5), ("doc2", 11.0)]; // BM25-like
let list2 = vec![("doc2", 0.9), ("doc3", 0.8)];   // Cosine similarity

// Z-score normalization with clipping
let fused = dbsf(&list1, &list2);
```

### Standardized Fusion (ERANK-style)

```rust
use rank_fusion::{standardized_with_config, StandardizedConfig};

let list1 = vec![("doc1", 12.5), ("doc2", 11.0)];
let list2 = vec![("doc2", 0.9), ("doc3", 0.8)];

// Z-score with configurable clipping (default: [-3, 3])
let config = StandardizedConfig::new((-3.0, 3.0)).with_top_k(10);
let fused = standardized_with_config(&list1, &list2, config);
```

### Additive Multi-Task Fusion

```rust
use rank_fusion::{additive_multi_task_with_config, AdditiveMultiTaskConfig, Normalization};

// E-commerce: CTR + CTCVR
let ctr_scores = vec![("product1", 0.05), ("product2", 0.03)];
let ctcvr_scores = vec![("product2", 0.02), ("product3", 0.01)];

// Conversion is 20× more important than click
let config = AdditiveMultiTaskConfig::new((1.0, 20.0))
    .with_normalization(Normalization::MinMax);
let fused = additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config);
```

### Multi-List Fusion

```rust
use rank_fusion::rrf_multi;
use rank_fusion::RrfConfig;

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];
let sparse = vec![("doc1", 0.7), ("doc4", 0.6)];

// Fuse 3+ lists
let lists = vec![&bm25[..], &dense[..], &sparse[..]];
let fused = rrf_multi(&lists, RrfConfig::default());
```

---

## Error Handling Examples

### Handling Weighted Fusion Errors

```rust
use rank_fusion::{weighted_multi, FusionError};

let list1 = vec![("doc1", 0.9), ("doc2", 0.5)];
let list2 = vec![("doc2", 0.8), ("doc3", 0.3)];

// Weights sum to zero (invalid)
let lists = vec![(&list1[..], 0.0), (&list2[..], 0.0)];

match weighted_multi(&lists, true, None) {
    Ok(fused) => println!("Fused: {:?}", fused),
    Err(FusionError::ZeroWeights) => {
        eprintln!("Error: weights sum to zero");
        // Handle error: use default weights or skip fusion
    }
    Err(FusionError::InvalidConfig(msg)) => {
        eprintln!("Invalid config: {}", msg);
    }
}
```

### Validating Fusion Results

```rust
use rank_fusion::{rrf, validate, ValidationResult};

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];

let fused = rrf(&bm25, &dense);

// Validate results
let validation = validate(&fused, false, Some(10));

if !validation.is_valid {
    eprintln!("Validation errors:");
    for error in &validation.errors {
        eprintln!("  - {}", error);
    }
}

if !validation.warnings.is_empty() {
    eprintln!("Validation warnings:");
    for warning in &validation.warnings {
        eprintln!("  - {}", warning);
    }
}
```

### Handling Edge Cases

```rust
use rank_fusion::rrf;

// Empty lists
let empty: Vec<(&str, f32)> = vec![];
let list = vec![("doc1", 0.9)];

let fused = rrf(&empty, &list);
// Result: [("doc1", 0.016)] (only from non-empty list)

// Single-item lists
let single1 = vec![("doc1", 0.9)];
let single2 = vec![("doc2", 0.8)];

let fused = rrf(&single1, &single2);
// Result: [("doc1", 0.016), ("doc2", 0.016)] (equal scores, order preserved)

// All identical scores (still works, rank-based)
let list1 = vec![("doc1", 0.5), ("doc2", 0.5)];
let list2 = vec![("doc2", 0.5), ("doc3", 0.5)];

let fused = rrf(&list1, &list2);
// Result: [("doc2", 0.033), ("doc1", 0.016), ("doc3", 0.016)]
// doc2 wins because it appears in both lists (consensus)
```

---

## Real-World Integration Examples

### Elasticsearch Integration

```rust
use rank_fusion::rrf;
use serde_json::Value;

// Simulate Elasticsearch BM25 results
fn get_bm25_results(query: &str) -> Vec<(String, f32)> {
    // In real code, this would call Elasticsearch
    vec![
        ("doc_123".to_string(), 12.5),
        ("doc_456".to_string(), 11.0),
        ("doc_789".to_string(), 9.5),
    ]
}

// Simulate dense vector search results
fn get_dense_results(query: &str) -> Vec<(String, f32)> {
    // In real code, this would call vector DB (Qdrant, Pinecone, etc.)
    vec![
        ("doc_456".to_string(), 0.92),
        ("doc_123".to_string(), 0.88),
        ("doc_999".to_string(), 0.85),
    ]
}

// Hybrid search: combine BM25 + dense
fn hybrid_search(query: &str) -> Vec<(String, f32)> {
    let bm25 = get_bm25_results(query);
    let dense = get_dense_results(query);
    
    // RRF finds consensus across retrievers
    rrf(&bm25, &dense)
}

// Usage
let results = hybrid_search("machine learning");
for (doc_id, score) in results.iter().take(10) {
    println!("{}: {:.4}", doc_id, score);
}
```

### Qdrant Vector DB Integration

```rust
use rank_fusion::rrf;

// Simulate Qdrant dense search
async fn qdrant_search(query_embedding: &[f32], limit: usize) -> Vec<(String, f32)> {
    // In real code:
    // let client = QdrantClient::from_url("http://localhost:6333").await?;
    // let results = client.search_points(...).await?;
    
    vec![
        ("doc_456".to_string(), 0.92),
        ("doc_123".to_string(), 0.88),
    ]
}

// Simulate BM25 search (Elasticsearch, Meilisearch, etc.)
async fn bm25_search(query: &str, limit: usize) -> Vec<(String, f32)> {
    vec![
        ("doc_123".to_string(), 12.5),
        ("doc_456".to_string(), 11.0),
    ]
}

// Hybrid search combining both
async fn hybrid_search_async(query: &str, query_embedding: &[f32]) -> Vec<(String, f32)> {
    let (bm25, dense) = tokio::join!(
        bm25_search(query, 50),
        qdrant_search(query_embedding, 50)
    );
    
    rrf(&bm25, &dense)
}
```

### LangChain Integration

```python
# Python example (would need Python bindings)
from langchain.retrievers import BM25Retriever, VectorStoreRetriever
import rank_fusion

# Initialize retrievers
bm25_retriever = BM25Retriever.from_documents(documents)
vector_retriever = VectorStoreRetriever(vectorstore=vectorstore)

# Hybrid retrieval
def hybrid_retrieve(query: str, k: int = 10):
    # Get results from both retrievers
    bm25_results = [(doc.page_content, score) 
                    for doc, score in bm25_retriever.get_relevant_documents_with_scores(query)]
    vector_results = [(doc.page_content, score)
                     for doc, score in vector_retriever.get_relevant_documents_with_scores(query)]
    
    # Fuse with RRF
    fused = rank_fusion.rrf(bm25_results, vector_results, k=60)
    
    # Return top k
    return [doc_id for doc_id, score in fused[:k]]
```

### E-commerce Multi-Task Ranking

```rust
use rank_fusion::{additive_multi_task_with_config, AdditiveMultiTaskConfig, Normalization};

// Simulate CTR scores (click-through rate)
fn get_ctr_scores(product_ids: &[String]) -> Vec<(String, f32)> {
    // In real code, query analytics database
    product_ids.iter()
        .map(|id| (id.clone(), 0.05)) // Example: 5% CTR
        .collect()
}

// Simulate CTCVR scores (click-to-conversion rate)
fn get_ctcvr_scores(product_ids: &[String]) -> Vec<(String, f32)> {
    // In real code, query analytics database
    product_ids.iter()
        .map(|id| (id.clone(), 0.02)) // Example: 2% CTCVR
        .collect()
}

// Multi-task ranking: optimize both CTR and conversion
fn rank_products(product_ids: &[String]) -> Vec<(String, f32)> {
    let ctr_scores = get_ctr_scores(product_ids);
    let ctcvr_scores = get_ctcvr_scores(product_ids);
    
    // Conversion is 20× more important than clicks
    let config = AdditiveMultiTaskConfig::new((1.0, 20.0))
        .with_normalization(Normalization::MinMax)
        .with_top_k(20);
    
    additive_multi_task_with_config(&ctr_scores, &ctcvr_scores, config)
}
```

---

## Explainability Examples

### Debugging Fusion Results

```rust
use rank_fusion::rrf_explain;
use rank_fusion::explain::RetrieverId;

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];

let lists = vec![&bm25[..], &dense[..]];
let retriever_ids = vec![
    RetrieverId::new("bm25"),
    RetrieverId::new("dense"),
];

let explained = rrf_explain(&lists, &retriever_ids, Default::default());

for result in &explained {
    println!("Document: {}", result.id);
    println!("  Final score: {:.4}", result.score);
    println!("  Rank: {}", result.rank);
    println!("  Consensus: {:.2}%", result.explanation.consensus_score * 100.0);
    
    for source in &result.explanation.sources {
        println!("  From {}:", source.retriever_id);
        if let Some(rank) = source.original_rank {
            println!("    Original rank: {}", rank);
        }
        if let Some(score) = source.original_score {
            println!("    Original score: {:.4}", score);
        }
        println!("    Contribution: {:.4}", source.contribution);
    }
    println!();
}
```

### Building User-Facing Explanations

```rust
use rank_fusion::rrf_explain;
use rank_fusion::explain::RetrieverId;

fn explain_result_to_user(result: &rank_fusion::explain::FusedResult<String>) -> String {
    let mut explanation = format!(
        "This result ranked #{} because it ",
        result.rank + 1
    );
    
    if result.explanation.consensus_score >= 0.8 {
        explanation.push_str("appeared in multiple search methods, indicating strong relevance.");
    } else if result.explanation.sources.len() == 1 {
        explanation.push_str(&format!(
            "matched well in {} search.",
            result.explanation.sources[0].retriever_id
        ));
    } else {
        explanation.push_str("matched across different search methods.");
    }
    
    explanation
}

// Usage
let lists = vec![&bm25[..], &dense[..]];
let retriever_ids = vec![
    RetrieverId::new("keyword_search"),
    RetrieverId::new("semantic_search"),
];

let explained = rrf_explain(&lists, &retriever_ids, Default::default());

for result in &explained {
    println!("{}", explain_result_to_user(result));
}
```

### Identifying Retriever Disagreements

```rust
use rank_fusion::rrf_explain;
use rank_fusion::explain::{RetrieverId, ConsensusReport};

fn analyze_consensus(explained: &[rank_fusion::explain::FusedResult<String>]) -> ConsensusReport<String> {
    let mut high_consensus = Vec::new();
    let mut single_source = Vec::new();
    let mut rank_disagreement = Vec::new();
    
    for result in explained {
        if result.explanation.consensus_score >= 0.8 {
            high_consensus.push(result.id.clone());
        } else if result.explanation.sources.len() == 1 {
            single_source.push(result.id.clone());
        } else {
            // Check for rank disagreement
            let ranks: Vec<_> = result.explanation.sources
                .iter()
                .filter_map(|s| s.original_rank)
                .collect();
            
            if ranks.iter().max().unwrap() - ranks.iter().min().unwrap() > 5 {
                rank_disagreement.push((
                    result.id.clone(),
                    result.explanation.sources.iter()
                        .map(|s| (s.retriever_id.clone(), s.original_rank.unwrap_or(0)))
                        .collect(),
                ));
            }
        }
    }
    
    ConsensusReport {
        high_consensus,
        single_source,
        rank_disagreement,
    }
}
```

---

## Validation Examples

### Comprehensive Validation

```rust
use rank_fusion::{rrf, validate, validate_sorted, validate_no_duplicates, 
                  validate_finite_scores, validate_bounds};

let bm25 = vec![("doc1", 0.9), ("doc2", 0.5)];
let dense = vec![("doc2", 0.8), ("doc3", 0.3)];

let fused = rrf(&bm25, &dense);

// Individual validations
let sorted_check = validate_sorted(&fused);
assert!(sorted_check.is_valid, "Results must be sorted");

let duplicates_check = validate_no_duplicates(&fused);
assert!(duplicates_check.is_valid, "No duplicate document IDs");

let finite_check = validate_finite_scores(&fused);
assert!(finite_check.is_valid, "All scores must be finite");

let bounds_check = validate_bounds(&fused, Some(10));
assert!(bounds_check.is_valid, "Results within expected bounds");

// Comprehensive validation
let full_check = validate(&fused, false, Some(10));
if !full_check.is_valid {
    panic!("Validation failed: {:?}", full_check.errors);
}
```

### Handling Validation Errors

```rust
use rank_fusion::{rrf, validate, ValidationResult};

fn safe_fusion(
    list1: &[(String, f32)],
    list2: &[(String, f32)],
) -> Result<Vec<(String, f32)>, String> {
    let fused = rrf(list1, list2);
    
    let validation = validate(&fused, true, None);
    
    if !validation.is_valid {
        let errors = validation.errors.join(", ");
        return Err(format!("Fusion validation failed: {}", errors));
    }
    
    if !validation.warnings.is_empty() {
        eprintln!("Warnings: {:?}", validation.warnings);
    }
    
    Ok(fused)
}
```

---

## Advanced Configuration Examples

### Dynamic Algorithm Selection

```rust
use rank_fusion::FusionMethod;

fn select_fusion_method(score_scales_compatible: bool, want_consensus: bool) -> FusionMethod {
    if !score_scales_compatible {
        FusionMethod::Rrf { k: 60 }
    } else if want_consensus {
        FusionMethod::CombMnz
    } else {
        FusionMethod::CombSum
    }
}

// Usage
let method = select_fusion_method(false, false);
let fused = method.fuse(&list1, &list2);
```

### Query-Dependent Fusion

```rust
use rank_fusion::{rrf, combsum, RrfConfig};

fn adaptive_fusion(
    query: &str,
    bm25: &[(String, f32)],
    dense: &[(String, f32)],
) -> Vec<(String, f32)> {
    // Keyword-heavy queries: trust BM25 more
    if query.split_whitespace().count() > 5 {
        // Use RRF (rank-based, no score bias)
        rrf(bm25, dense)
    } else {
        // Semantic queries: use score-based (assumes compatible scales)
        combsum(bm25, dense)
    }
}
```

### Batch Processing

```rust
use rank_fusion::rrf;

fn batch_fusion(
    queries: &[String],
    bm25_results: &[Vec<(String, f32)>],
    dense_results: &[Vec<(String, f32)>],
) -> Vec<Vec<(String, f32)>> {
    queries.iter()
        .zip(bm25_results.iter().zip(dense_results.iter()))
        .map(|(_, (bm25, dense))| rrf(bm25, dense))
        .collect()
}
```

---

## Python Examples

### Basic Usage

```python
import rank_fusion

# RRF fusion
bm25 = [("doc1", 12.5), ("doc2", 11.0)]
dense = [("doc2", 0.9), ("doc3", 0.8)]

fused = rank_fusion.rrf(bm25, dense, k=60)
print(fused)  # [("doc2", 0.033), ("doc1", 0.016), ("doc3", 0.016)]
```

### Error Handling

```python
import rank_fusion

try:
    # Invalid k (would cause division by zero)
    fused = rank_fusion.rrf(bm25, dense, k=0)
except ValueError as e:
    print(f"Error: {e}")
```

### Validation

```python
import rank_fusion

fused = rank_fusion.rrf(bm25, dense)

# Validate results
result = rank_fusion.validate(fused, check_non_negative=False, max_results=10)

if not result.is_valid:
    print("Errors:", result.errors)
if result.warnings:
    print("Warnings:", result.warnings)
```

### Explainability

```python
import rank_fusion

lists = [bm25, dense]
retriever_ids = ["bm25", "dense"]

explained = rank_fusion.rrf_explain(lists, retriever_ids, k=60)

for result in explained:
    print(f"Document: {result.id}")
    print(f"  Score: {result.score:.4}")
    print(f"  Rank: {result.rank}")
    print(f"  Consensus: {result.explanation.consensus_score:.2%}")
    for source in result.explanation.sources:
        print(f"    {source.retriever_id}: contribution {source.contribution:.4}")
```

---

## Common Patterns

### Hybrid Search Pipeline

```rust
use rank_fusion::rrf;

fn hybrid_search_pipeline(query: &str) -> Vec<(String, f32)> {
    // Stage 1: Retrieve from multiple sources
    let bm25_results = retrieve_bm25(query, 50);
    let dense_results = retrieve_dense(query, 50);
    let sparse_results = retrieve_sparse(query, 50);
    
    // Stage 2: Fuse results
    let lists = vec![&bm25_results[..], &dense_results[..], &sparse_results[..]];
    let fused = rrf_multi(&lists, Default::default());
    
    // Stage 3: Validate
    let validation = validate(&fused, false, Some(20));
    if !validation.is_valid {
        eprintln!("Validation errors: {:?}", validation.errors);
    }
    
    // Stage 4: Return top results
    fused.into_iter().take(20).collect()
}
```

### A/B Testing Fusion Methods

```rust
use rank_fusion::{rrf, combsum};

fn ab_test_fusion(
    method: &str,
    list1: &[(String, f32)],
    list2: &[(String, f32)],
) -> Vec<(String, f32)> {
    match method {
        "rrf" => rrf(list1, list2),
        "combsum" => combsum(list1, list2),
        _ => panic!("Unknown method: {}", method),
    }
}

// Evaluate both methods
let rrf_results = ab_test_fusion("rrf", &list1, &list2);
let combsum_results = ab_test_fusion("combsum", &list1, &list2);

// Compare (would use rank-eval for metrics)
println!("RRF top 5: {:?}", &rrf_results[..5]);
println!("CombSUM top 5: {:?}", &combsum_results[..5]);
```

