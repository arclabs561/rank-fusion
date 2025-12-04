//! Dataset loaders for MS MARCO, BEIR, TREC, MIRACL, MTEB, and other IR datasets.
//!
//! **Note:** This module re-exports functionality from `rank-eval::dataset` for backward compatibility.
//! New code should use `rank_eval::dataset` directly.

// Re-export everything from rank-eval::dataset
#[allow(unused_imports)] // Re-exports for external use
pub use rank_eval::dataset::*;
