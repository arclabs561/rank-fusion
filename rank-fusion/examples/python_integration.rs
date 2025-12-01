//! Example: How rank-fusion would be used from Python (conceptual).
//!
//! This demonstrates the Python API design, though actual Python bindings
//! require proper setup with maturin or setuptools-rust.
//!
//! # Python Usage (conceptual)
//!
//! ```python
//! import rank_fusion
//!
//! # Two-list fusion
//! bm25 = [("d1", 12.5), ("d2", 11.0)]
//! dense = [("d2", 0.9), ("d3", 0.8)]
//!
//! fused = rank_fusion.rrf(bm25, dense, k=60)
//! # [("d2", 0.033), ("d1", 0.016), ("d3", 0.016)]
//!
//! # Multi-list fusion
//! lists = [
//!     [("d1", 10.0), ("d2", 9.0)],
//!     [("d2", 0.9), ("d3", 0.8)],
//!     [("d1", 0.95), ("d3", 0.85)],
//! ]
//! fused = rank_fusion.rrf_multi(lists, k=60)
//!
//! # With explainability (when implemented)
//! explained = rank_fusion.rrf_explain(
//!     lists,
//!     retriever_ids=["bm25", "dense", "sparse"],
//!     k=60
//! )
//! for result in explained:
//!     print(f"{result.id}: {result.score}")
//!     print(f"  Consensus: {result.explanation.consensus_score:.2%}")
//!     for source in result.explanation.sources:
//!         print(f"    {source.retriever_id}: {source.contribution:.6}")
//! ```

fn main() {
    println!("This is a conceptual example of Python integration.");
    println!("See the source code comments for Python API design.");
    println!("\nTo build actual Python bindings:");
    println!("1. Install maturin: pip install maturin");
    println!("2. Create pyproject.toml with maturin configuration");
    println!("3. Build: maturin develop");
    println!("4. Use from Python: import rank_fusion");
}
