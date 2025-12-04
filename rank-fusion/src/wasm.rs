//! WebAssembly bindings for rank-fusion.
//!
//! This module provides JavaScript-compatible bindings for the rank-fusion algorithms.
//! It's only compiled when the `wasm` feature is enabled.
//!
//! # Usage Example
//!
//! ```javascript
//! import init, { rrf, combsum, rrf_multi, combsum_multi } from '@arclabs561/rank-fusion';
//!
//! await init();
//!
//! const bm25 = [["d1", 12.5], ["d2", 11.0]];
//! const dense = [["d2", 0.9], ["d3", 0.8]];
//!
//! // RRF fusion (two lists)
//! const fused = rrf(bm25, dense, 60);
//!
//! // CombSUM fusion (two lists)
//! const fused2 = combsum(bm25, dense);
//!
//! // RRF fusion (multiple lists)
//! const lists = [bm25, dense, [["d1", 0.5], ["d4", 0.3]]];
//! const fused3 = rrf_multi(lists, 60);
//!
//! // CombSUM fusion (multiple lists) with top_k
//! const fused4 = combsum_multi(lists, 10); // top 10 results
//! ```
//!
//! # Input Format
//!
//! All functions expect arrays of `[id, score]` pairs where:
//! - `id` is a string (document identifier)
//! - `score` is a finite number (ranking score)
//!
//! # Available Functions
//!
//! ## Two-List Fusion
//! - `rrf(results_a, results_b, k?, top_k?)` - Reciprocal Rank Fusion
//! - `isr(results_a, results_b, k?, top_k?)` - Inverse Square Rank
//! - `combsum(results_a, results_b)` - CombSUM fusion
//! - `combmnz(results_a, results_b)` - CombMNZ fusion
//! - `borda(results_a, results_b)` - Borda count
//! - `dbsf(results_a, results_b)` - Dense Boolean Score Fusion
//! - `weighted(results_a, results_b, weight_a, weight_b, normalize, top_k?)` - Weighted fusion
//! - `standardized(results_a, results_b, clip_min?, clip_max?, top_k?)` - ERANK-style z-score fusion
//! - `additive_multi_task(results_a, results_b, weight_a?, weight_b?, normalization?, top_k?)` - ResFlow-style fusion
//!
//! ## Multi-List Fusion
//! - `rrf_multi(lists, k?, top_k?)` - RRF for 3+ lists
//! - `isr_multi(lists, k?, top_k?)` - ISR for 3+ lists
//! - `combsum_multi(lists, top_k?)` - CombSUM for 3+ lists
//! - `combmnz_multi(lists, top_k?)` - CombMNZ for 3+ lists
//! - `borda_multi(lists, top_k?)` - Borda count for 3+ lists
//! - `dbsf_multi(lists, top_k?)` - DBSF for 3+ lists
//! - `standardized_multi(lists, clip_min?, clip_max?, top_k?)` - Standardized fusion for 3+ lists
//!
//! ## Explainability
//! - `rrf_explain(lists, retriever_ids, k?, top_k?)` - RRF with provenance
//! - `combsum_explain(lists, retriever_ids, top_k?)` - CombSUM with provenance
//! - `combmnz_explain(lists, retriever_ids, top_k?)` - CombMNZ with provenance
//! - `dbsf_explain(lists, retriever_ids, top_k?)` - DBSF with provenance
//!
//! ## Validation
//! - `validate(results, strict, expected_top_k?)` - Comprehensive validation
//! - `validate_sorted(results)` - Check if sorted by score
//! - `validate_no_duplicates(results)` - Check for duplicate IDs
//! - `validate_finite_scores(results)` - Check for NaN/Infinity
//! - `validate_non_negative_scores(results)` - Warn on negative scores
//! - `validate_bounds(results, max_results?)` - Check result count
//!
//! # Error Handling
//!
//! All functions return `Result<JsValue, JsValue>`. Errors include:
//! - Invalid input format (not arrays, wrong types)
//! - Invalid parameters (k=0 for RRF/ISR, zero weights for weighted fusion)
//! - Non-finite scores (NaN, Infinity)
//! - Empty input arrays return empty results (not errors)

#[cfg(feature = "wasm")]
use wasm_bindgen::prelude::*;

#[cfg(feature = "wasm")]
use crate::{
    additive_multi_task_with_config, standardized_with_config, AdditiveMultiTaskConfig,
    FusionConfig, Normalization, RrfConfig, StandardizedConfig, WeightedConfig,
};

/// Helper to convert JS array of [id, score] pairs to Vec<(String, f32)>
///
/// This function efficiently converts JavaScript arrays to Rust vectors with validation.
/// Uses direct memory access patterns for better performance.
#[cfg(feature = "wasm")]
fn js_to_results(js: &JsValue) -> Result<Vec<(String, f32)>, JsValue> {
    use wasm_bindgen::JsCast;

    let array = js
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array"))?;

    // Pre-allocate with estimated capacity for better performance
    let len = array.length() as usize;
    let mut results = Vec::with_capacity(len);

    for (idx, item) in array.iter().enumerate() {
        let pair = item.dyn_ref::<js_sys::Array>().ok_or_else(|| {
            JsValue::from_str(&format!("Expected [id, score] pair at index {}", idx))
        })?;
        if pair.length() != 2 {
            return Err(JsValue::from_str(&format!(
                "Expected [id, score] pair at index {}, got array of length {}",
                idx,
                pair.length()
            )));
        }
        let id = pair
            .get(0)
            .as_string()
            .ok_or_else(|| JsValue::from_str(&format!("id must be a string at index {}", idx)))?;
        let score_val = pair.get(1).as_f64().ok_or_else(|| {
            JsValue::from_str(&format!("score must be a number at index {}", idx))
        })?;

        // Validate score is finite (not NaN or Infinity)
        if !score_val.is_finite() {
            return Err(JsValue::from_str(&format!(
                "score must be a finite number at index {}, got {}",
                idx, score_val
            )));
        }

        results.push((id, score_val as f32));
    }
    Ok(results)
}

/// Helper to convert Vec<(String, f32)> to JS array of [id, score] pairs
#[cfg(feature = "wasm")]
fn results_to_js(results: &[(String, f32)]) -> JsValue {
    let array = js_sys::Array::new();
    for (id, score) in results {
        let pair = js_sys::Array::new();
        pair.push(&JsValue::from_str(id));
        pair.push(&JsValue::from_f64(*score as f64));
        array.push(&pair);
    }
    array.into()
}

/// Reciprocal Rank Fusion (RRF) for two result lists.
///
/// # Arguments
/// * `results_a` - First result list as array of `[id, score]` pairs
/// * `results_b` - Second result list as array of `[id, score]` pairs
/// * `k` - Smoothing constant (default: 60, must be >= 1)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
///
/// # Errors
/// Returns error if `k` is 0 (would cause division by zero) or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rrf(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };

    let fused = crate::rrf_with_config(&a, &b, config);
    Ok(results_to_js(&fused))
}

/// Inverse Square Rank (ISR) for two result lists.
///
/// # Arguments
/// * `results_a` - First result list as array of `[id, score]` pairs
/// * `results_b` - Second result list as array of `[id, score]` pairs
/// * `k` - Smoothing constant (default: 1, must be >= 1)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Errors
/// Returns error if `k` is 0 (would cause division by zero) or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn isr(
    results_a: &JsValue,
    results_b: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let k_val = k.unwrap_or(1);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };

    let fused = crate::isr_with_config(&a, &b, config);
    Ok(results_to_js(&fused))
}

/// CombSUM fusion for two result lists.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn combsum(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    let fused = crate::combsum(&a, &b);
    Ok(results_to_js(&fused))
}

/// CombMNZ fusion for two result lists.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn combmnz(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    let fused = crate::combmnz(&a, &b);
    Ok(results_to_js(&fused))
}

/// Borda count fusion for two result lists.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn borda(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    let fused = crate::borda(&a, &b);
    Ok(results_to_js(&fused))
}

/// DBSF (Dense Boolean Score Fusion) for two result lists.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dbsf(results_a: &JsValue, results_b: &JsValue) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;
    let fused = crate::dbsf(&a, &b);
    Ok(results_to_js(&fused))
}

/// Weighted fusion for two result lists.
///
/// # Arguments
/// * `results_a` - First result list
/// * `results_b` - Second result list
/// * `weight_a` - Weight for first list (0.0 to 1.0)
/// * `weight_b` - Weight for second list (0.0 to 1.0)
/// * `normalize` - Whether to normalize scores before weighting
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Errors
/// Returns error if weights are invalid (NaN, Infinity, or both zero).
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn weighted(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: f32,
    weight_b: f32,
    normalize: bool,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    // Validate weights are finite
    if !weight_a.is_finite() || !weight_b.is_finite() {
        return Err(JsValue::from_str("weights must be finite numbers"));
    }

    // Check for zero weights (will cause error in weighted fusion)
    if (weight_a.abs() < 1e-9) && (weight_b.abs() < 1e-9) {
        return Err(JsValue::from_str("weights cannot both be zero"));
    }

    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let config = WeightedConfig {
        weight_a,
        weight_b,
        normalize,
        top_k,
    };

    let fused = crate::weighted(&a, &b, config);
    Ok(results_to_js(&fused))
}

/// RRF for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `k` - Smoothing constant (default: 60, must be >= 1)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Errors
/// Returns error if `k` is 0 (would cause division by zero) or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rrf_multi(
    lists: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };

    let fused = crate::rrf_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// Standardized fusion (ERANK-style) for two result lists.
///
/// Uses z-score normalization with configurable clipping to handle different score distributions.
///
/// # Arguments
/// * `results_a` - First result list as array of `[id, score]` pairs
/// * `results_b` - Second result list as array of `[id, score]` pairs
/// * `clip_min` - Minimum z-score clipping value (default: -3.0)
/// * `clip_max` - Maximum z-score clipping value (default: 3.0)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn standardized(
    results_a: &JsValue,
    results_b: &JsValue,
    clip_min: Option<f32>,
    clip_max: Option<f32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let clip_min = clip_min.unwrap_or(-3.0);
    let clip_max = clip_max.unwrap_or(3.0);

    let config = StandardizedConfig::new((clip_min, clip_max));
    let config = if let Some(k) = top_k {
        config.with_top_k(k)
    } else {
        config
    };

    let fused = standardized_with_config(&a, &b, config);
    Ok(results_to_js(&fused))
}

/// Additive multi-task fusion (ResFlow-style) for two result lists.
///
/// Combines scores from multiple tasks with configurable weights and normalization.
/// Optimized for e-commerce ranking (e.g., CTR + CTCVR).
///
/// # Arguments
/// * `results_a` - First result list as array of `[id, score]` pairs
/// * `results_b` - Second result list as array of `[id, score]` pairs
/// * `weight_a` - Weight for first task (default: 1.0)
/// * `weight_b` - Weight for second task (default: 1.0)
/// * `normalization` - Normalization method: "zscore", "minmax", "sum", "rank", "none" (default: "minmax")
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
///
/// # Errors
/// Returns error if normalization string is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn additive_multi_task(
    results_a: &JsValue,
    results_b: &JsValue,
    weight_a: Option<f32>,
    weight_b: Option<f32>,
    normalization: Option<String>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let a = js_to_results(results_a)?;
    let b = js_to_results(results_b)?;

    let weight_a = weight_a.unwrap_or(1.0);
    let weight_b = weight_b.unwrap_or(1.0);

    let norm_str = normalization.as_deref().unwrap_or("minmax");
    let norm = match norm_str.to_lowercase().as_str() {
        "zscore" => Normalization::ZScore,
        "minmax" => Normalization::MinMax,
        "sum" => Normalization::Sum,
        "rank" => Normalization::Rank,
        "none" => Normalization::None,
        _ => {
            return Err(JsValue::from_str(
                "normalization must be one of: zscore, minmax, sum, rank, none",
            ))
        }
    };

    let config = AdditiveMultiTaskConfig::new((weight_a, weight_b)).with_normalization(norm);
    let config = if let Some(k) = top_k {
        config.with_top_k(k)
    } else {
        config
    };

    let fused = additive_multi_task_with_config(&a, &b, config);
    Ok(results_to_js(&fused))
}

/// ISR for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `k` - Smoothing constant (default: 1, must be >= 1)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Errors
/// Returns error if `k` is 0 (would cause division by zero) or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn isr_multi(
    lists: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let k_val = k.unwrap_or(1);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };

    let fused = crate::isr_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// CombSUM fusion for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn combsum_multi(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let config = FusionConfig { top_k };
    let fused = crate::combsum_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// CombMNZ fusion for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn combmnz_multi(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let config = FusionConfig { top_k };
    let fused = crate::combmnz_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// Borda count fusion for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn borda_multi(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let config = FusionConfig { top_k };
    let fused = crate::borda_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// DBSF (Dense Boolean Score Fusion) for multiple result lists.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn dbsf_multi(lists: &JsValue, top_k: Option<usize>) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let config = FusionConfig { top_k };
    let fused = crate::dbsf_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// Standardized fusion (ERANK-style) for multiple result lists.
///
/// Uses z-score normalization with configurable clipping to handle different score distributions.
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `clip_min` - Minimum z-score clipping value (default: -3.0)
/// * `clip_max` - Maximum z-score clipping value (default: 3.0)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Fused results as array of `[id, score]` pairs, sorted by score descending.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn standardized_multi(
    lists: &JsValue,
    clip_min: Option<f32>,
    clip_max: Option<f32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let clip_min = clip_min.unwrap_or(-3.0);
    let clip_max = clip_max.unwrap_or(3.0);

    let config = StandardizedConfig::new((clip_min, clip_max));
    let config = if let Some(k) = top_k {
        config.with_top_k(k)
    } else {
        config
    };

    let fused = crate::standardized_multi(&lists_vec, config);
    Ok(results_to_js(&fused))
}

/// RRF fusion with explainability (returns provenance information).
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `retriever_ids` - Array of retriever identifiers (strings)
/// * `k` - Smoothing constant (default: 60, must be >= 1)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Array of objects with structure:
/// ```javascript
/// {
///   id: "doc1",
///   score: 0.123,
///   explanation: {
///     sources: [
///       { retriever_id: "bm25", rank: 0, score: 12.5, contribution: 0.016 },
///       { retriever_id: "dense", rank: 1, score: 0.9, contribution: 0.016 }
///     ],
///     consensus_score: 1.0, // fraction of lists containing this doc
///     total_contributions: 0.032
///   }
/// }
/// ```
///
/// # Errors
/// Returns error if `k` is 0, if retriever_ids length doesn't match lists, or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn rrf_explain(
    lists: &JsValue,
    retriever_ids: &JsValue,
    k: Option<u32>,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let ids_array = retriever_ids
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of retriever IDs"))?;

    if ids_array.length() as usize != lists_vec.len() {
        return Err(JsValue::from_str(&format!(
            "retriever_ids length ({}) must match lists length ({})",
            ids_array.length(),
            lists_vec.len()
        )));
    }

    let mut retriever_ids_vec = Vec::new();
    for item in ids_array.iter() {
        let id = item
            .as_string()
            .ok_or_else(|| JsValue::from_str("retriever_id must be a string"))?;
        retriever_ids_vec.push(crate::RetrieverId::new(&id));
    }

    let k_val = k.unwrap_or(60);
    if k_val == 0 {
        return Err(JsValue::from_str(
            "k must be >= 1 to avoid division by zero",
        ));
    }

    let config = RrfConfig { k: k_val, top_k };
    let explained = crate::rrf_explain(&lists_vec, &retriever_ids_vec, config);

    // Convert to JS objects
    let result = js_sys::Array::new();
    for fused_result in explained.iter() {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&fused_result.id))?;
        js_sys::Reflect::set(
            &obj,
            &"score".into(),
            &JsValue::from_f64(fused_result.score as f64),
        )?;

        // Build explanation object
        let explanation = js_sys::Object::new();
        let sources = js_sys::Array::new();
        for source in fused_result.explanation.sources.iter() {
            let source_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &source_obj,
                &"retriever_id".into(),
                &JsValue::from_str(&source.retriever_id),
            )?;
            if let Some(rank) = source.original_rank {
                js_sys::Reflect::set(
                    &source_obj,
                    &"rank".into(),
                    &JsValue::from_f64(rank as f64),
                )?;
            }
            if let Some(score) = source.original_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"score".into(),
                    &JsValue::from_f64(score as f64),
                )?;
            }
            if let Some(norm_score) = source.normalized_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"normalized_score".into(),
                    &JsValue::from_f64(norm_score as f64),
                )?;
            }
            js_sys::Reflect::set(
                &source_obj,
                &"contribution".into(),
                &JsValue::from_f64(source.contribution as f64),
            )?;
            sources.push(&source_obj);
        }
        js_sys::Reflect::set(&explanation, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &explanation,
            &"consensus_score".into(),
            &JsValue::from_f64(fused_result.explanation.consensus_score as f64),
        )?;
        js_sys::Reflect::set(
            &explanation,
            &"total_contributions".into(),
            &JsValue::from_f64(
                fused_result
                    .explanation
                    .sources
                    .iter()
                    .map(|s| s.contribution)
                    .sum::<f32>() as f64,
            ),
        )?;
        js_sys::Reflect::set(&obj, &"explanation".into(), &explanation)?;
        result.push(&obj);
    }

    Ok(result.into())
}

/// CombSUM fusion with explainability (returns provenance information).
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `retriever_ids` - Array of retriever identifiers (strings)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Array of objects with same structure as `rrf_explain`.
///
/// # Errors
/// Returns error if retriever_ids length doesn't match lists, or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = combsumExplain)]
pub fn combsum_explain(
    lists: &JsValue,
    retriever_ids: &JsValue,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let ids_array = retriever_ids
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of retriever IDs"))?;

    if ids_array.length() as usize != lists_vec.len() {
        return Err(JsValue::from_str(&format!(
            "retriever_ids length ({}) must match lists length ({})",
            ids_array.length(),
            lists_vec.len()
        )));
    }

    let mut retriever_ids_vec = Vec::new();
    for item in ids_array.iter() {
        let id = item
            .as_string()
            .ok_or_else(|| JsValue::from_str("retriever_id must be a string"))?;
        retriever_ids_vec.push(crate::RetrieverId::new(&id));
    }

    let config = crate::FusionConfig { top_k };
    let explained = crate::combsum_explain(&lists_vec, &retriever_ids_vec, config);

    // Convert to JS objects (same structure as rrf_explain)
    let result = js_sys::Array::new();
    for fused_result in explained.iter() {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&fused_result.id))?;
        js_sys::Reflect::set(
            &obj,
            &"score".into(),
            &JsValue::from_f64(fused_result.score as f64),
        )?;

        let explanation = js_sys::Object::new();
        let sources = js_sys::Array::new();
        for source in fused_result.explanation.sources.iter() {
            let source_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &source_obj,
                &"retriever_id".into(),
                &JsValue::from_str(&source.retriever_id),
            )?;
            if let Some(rank) = source.original_rank {
                js_sys::Reflect::set(
                    &source_obj,
                    &"rank".into(),
                    &JsValue::from_f64(rank as f64),
                )?;
            }
            if let Some(score) = source.original_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"score".into(),
                    &JsValue::from_f64(score as f64),
                )?;
            }
            if let Some(norm_score) = source.normalized_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"normalized_score".into(),
                    &JsValue::from_f64(norm_score as f64),
                )?;
            }
            js_sys::Reflect::set(
                &source_obj,
                &"contribution".into(),
                &JsValue::from_f64(source.contribution as f64),
            )?;
            sources.push(&source_obj);
        }
        js_sys::Reflect::set(&explanation, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &explanation,
            &"consensus_score".into(),
            &JsValue::from_f64(fused_result.explanation.consensus_score as f64),
        )?;
        js_sys::Reflect::set(&obj, &"explanation".into(), &explanation)?;
        result.push(&obj);
    }

    Ok(result.into())
}

/// CombMNZ fusion with explainability (returns provenance information).
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `retriever_ids` - Array of retriever identifiers (strings)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Array of objects with same structure as `rrf_explain`.
///
/// # Errors
/// Returns error if retriever_ids length doesn't match lists, or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = combmnzExplain)]
pub fn combmnz_explain(
    lists: &JsValue,
    retriever_ids: &JsValue,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let ids_array = retriever_ids
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of retriever IDs"))?;

    if ids_array.length() as usize != lists_vec.len() {
        return Err(JsValue::from_str(&format!(
            "retriever_ids length ({}) must match lists length ({})",
            ids_array.length(),
            lists_vec.len()
        )));
    }

    let mut retriever_ids_vec = Vec::new();
    for item in ids_array.iter() {
        let id = item
            .as_string()
            .ok_or_else(|| JsValue::from_str("retriever_id must be a string"))?;
        retriever_ids_vec.push(crate::RetrieverId::new(&id));
    }

    let config = crate::FusionConfig { top_k };
    let explained = crate::combmnz_explain(&lists_vec, &retriever_ids_vec, config);

    // Convert to JS objects (same structure as rrf_explain)
    let result = js_sys::Array::new();
    for fused_result in explained.iter() {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&fused_result.id))?;
        js_sys::Reflect::set(
            &obj,
            &"score".into(),
            &JsValue::from_f64(fused_result.score as f64),
        )?;

        let explanation = js_sys::Object::new();
        let sources = js_sys::Array::new();
        for source in fused_result.explanation.sources.iter() {
            let source_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &source_obj,
                &"retriever_id".into(),
                &JsValue::from_str(&source.retriever_id),
            )?;
            if let Some(rank) = source.original_rank {
                js_sys::Reflect::set(
                    &source_obj,
                    &"rank".into(),
                    &JsValue::from_f64(rank as f64),
                )?;
            }
            if let Some(score) = source.original_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"score".into(),
                    &JsValue::from_f64(score as f64),
                )?;
            }
            if let Some(norm_score) = source.normalized_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"normalized_score".into(),
                    &JsValue::from_f64(norm_score as f64),
                )?;
            }
            js_sys::Reflect::set(
                &source_obj,
                &"contribution".into(),
                &JsValue::from_f64(source.contribution as f64),
            )?;
            sources.push(&source_obj);
        }
        js_sys::Reflect::set(&explanation, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &explanation,
            &"consensus_score".into(),
            &JsValue::from_f64(fused_result.explanation.consensus_score as f64),
        )?;
        js_sys::Reflect::set(&obj, &"explanation".into(), &explanation)?;
        result.push(&obj);
    }

    Ok(result.into())
}

/// DBSF fusion with explainability (returns provenance information).
///
/// # Arguments
/// * `lists` - Array of result lists, each as array of `[id, score]` pairs
/// * `retriever_ids` - Array of retriever identifiers (strings)
/// * `top_k` - Maximum number of results to return (default: None = all)
///
/// # Returns
/// Array of objects with same structure as `rrf_explain`.
///
/// # Errors
/// Returns error if retriever_ids length doesn't match lists, or if input is invalid.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = dbsfExplain)]
pub fn dbsf_explain(
    lists: &JsValue,
    retriever_ids: &JsValue,
    top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    use wasm_bindgen::JsCast;

    let array = lists
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of lists"))?;

    if array.length() == 0 {
        return Ok(js_sys::Array::new().into());
    }

    let mut lists_vec = Vec::new();
    for item in array.iter() {
        let list = js_to_results(&item)?;
        lists_vec.push(list);
    }

    let ids_array = retriever_ids
        .dyn_ref::<js_sys::Array>()
        .ok_or_else(|| JsValue::from_str("Expected array of retriever IDs"))?;

    if ids_array.length() as usize != lists_vec.len() {
        return Err(JsValue::from_str(&format!(
            "retriever_ids length ({}) must match lists length ({})",
            ids_array.length(),
            lists_vec.len()
        )));
    }

    let mut retriever_ids_vec = Vec::new();
    for item in ids_array.iter() {
        let id = item
            .as_string()
            .ok_or_else(|| JsValue::from_str("retriever_id must be a string"))?;
        retriever_ids_vec.push(crate::RetrieverId::new(&id));
    }

    let config = crate::FusionConfig { top_k };
    let explained = crate::dbsf_explain(&lists_vec, &retriever_ids_vec, config);

    // Convert to JS objects (same structure as rrf_explain)
    let result = js_sys::Array::new();
    for fused_result in explained.iter() {
        let obj = js_sys::Object::new();
        js_sys::Reflect::set(&obj, &"id".into(), &JsValue::from_str(&fused_result.id))?;
        js_sys::Reflect::set(
            &obj,
            &"score".into(),
            &JsValue::from_f64(fused_result.score as f64),
        )?;

        let explanation = js_sys::Object::new();
        let sources = js_sys::Array::new();
        for source in fused_result.explanation.sources.iter() {
            let source_obj = js_sys::Object::new();
            js_sys::Reflect::set(
                &source_obj,
                &"retriever_id".into(),
                &JsValue::from_str(&source.retriever_id),
            )?;
            if let Some(rank) = source.original_rank {
                js_sys::Reflect::set(
                    &source_obj,
                    &"rank".into(),
                    &JsValue::from_f64(rank as f64),
                )?;
            }
            if let Some(score) = source.original_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"score".into(),
                    &JsValue::from_f64(score as f64),
                )?;
            }
            if let Some(norm_score) = source.normalized_score {
                js_sys::Reflect::set(
                    &source_obj,
                    &"normalized_score".into(),
                    &JsValue::from_f64(norm_score as f64),
                )?;
            }
            js_sys::Reflect::set(
                &source_obj,
                &"contribution".into(),
                &JsValue::from_f64(source.contribution as f64),
            )?;
            sources.push(&source_obj);
        }
        js_sys::Reflect::set(&explanation, &"sources".into(), &sources)?;
        js_sys::Reflect::set(
            &explanation,
            &"consensus_score".into(),
            &JsValue::from_f64(fused_result.explanation.consensus_score as f64),
        )?;
        js_sys::Reflect::set(&obj, &"explanation".into(), &explanation)?;
        result.push(&obj);
    }

    Ok(result.into())
}

/// Validate fusion results.
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
/// * `strict` - If true, treat warnings as errors
/// * `expected_top_k` - Expected number of results (None = no check)
///
/// # Returns
/// Validation result object:
/// ```javascript
/// {
///   is_valid: true,
///   errors: [],
///   warnings: [],
///   stats: {
///     total_results: 10,
///     has_duplicates: false,
///     is_sorted: true,
///     has_finite_scores: true,
///     has_non_negative_scores: true
///   }
/// }
/// ```
#[cfg(feature = "wasm")]
#[wasm_bindgen]
pub fn validate(
    results: &JsValue,
    strict: bool,
    expected_top_k: Option<usize>,
) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate(&results_vec, strict, expected_top_k);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let errors = js_sys::Array::new();
    for error in validation.errors.iter() {
        errors.push(&JsValue::from_str(error));
    }
    js_sys::Reflect::set(&obj, &"errors".into(), &errors)?;

    let warnings = js_sys::Array::new();
    for warning in validation.warnings.iter() {
        warnings.push(&JsValue::from_str(warning));
    }
    js_sys::Reflect::set(&obj, &"warnings".into(), &warnings)?;

    // Add basic stats computed from results
    let stats = js_sys::Object::new();
    js_sys::Reflect::set(
        &stats,
        &"total_results".into(),
        &JsValue::from_f64(results_vec.len() as f64),
    )?;
    
    // Check for duplicates
    let mut seen = std::collections::HashSet::new();
    let mut has_duplicates = false;
    for (id, _) in results_vec.iter() {
        if seen.contains(id) {
            has_duplicates = true;
            break;
        }
        seen.insert(id);
    }
    js_sys::Reflect::set(
        &stats,
        &"has_duplicates".into(),
        &JsValue::from_bool(has_duplicates),
    )?;
    
    // Check if sorted
    let mut is_sorted = true;
    for window in results_vec.windows(2) {
        if window[0].1 < window[1].1 {
            is_sorted = false;
            break;
        }
    }
    js_sys::Reflect::set(
        &stats,
        &"is_sorted".into(),
        &JsValue::from_bool(is_sorted),
    )?;
    
    // Check for finite scores
    let has_finite_scores = results_vec.iter().all(|(_, score)| score.is_finite());
    js_sys::Reflect::set(
        &stats,
        &"has_finite_scores".into(),
        &JsValue::from_bool(has_finite_scores),
    )?;
    
    // Check for non-negative scores
    let has_non_negative_scores = results_vec.iter().all(|(_, score)| *score >= 0.0);
    js_sys::Reflect::set(
        &stats,
        &"has_non_negative_scores".into(),
        &JsValue::from_bool(has_non_negative_scores),
    )?;
    
    js_sys::Reflect::set(&obj, &"stats".into(), &stats)?;

    Ok(obj.into())
}

/// Validate that fusion results are sorted by score (descending).
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
///
/// # Returns
/// Validation result object with `is_valid` boolean and `errors` array.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = validateSorted)]
pub fn validate_sorted(results: &JsValue) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate_sorted(&results_vec);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let errors = js_sys::Array::new();
    for error in validation.errors.iter() {
        errors.push(&JsValue::from_str(error));
    }
    js_sys::Reflect::set(&obj, &"errors".into(), &errors)?;

    Ok(obj.into())
}

/// Validate that fusion results contain no duplicate document IDs.
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
///
/// # Returns
/// Validation result object with `is_valid` boolean and `errors` array.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = validateNoDuplicates)]
pub fn validate_no_duplicates(results: &JsValue) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate_no_duplicates(&results_vec);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let errors = js_sys::Array::new();
    for error in validation.errors.iter() {
        errors.push(&JsValue::from_str(error));
    }
    js_sys::Reflect::set(&obj, &"errors".into(), &errors)?;

    Ok(obj.into())
}

/// Validate that all scores are finite (not NaN or Infinity).
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
///
/// # Returns
/// Validation result object with `is_valid` boolean and `errors` array.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = validateFiniteScores)]
pub fn validate_finite_scores(results: &JsValue) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate_finite_scores(&results_vec);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let errors = js_sys::Array::new();
    for error in validation.errors.iter() {
        errors.push(&JsValue::from_str(error));
    }
    js_sys::Reflect::set(&obj, &"errors".into(), &errors)?;

    Ok(obj.into())
}

/// Validate that all scores are non-negative (warning only).
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
///
/// # Returns
/// Validation result object with `is_valid` boolean and `warnings` array.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = validateNonNegativeScores)]
pub fn validate_non_negative_scores(results: &JsValue) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate_non_negative_scores(&results_vec);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let warnings = js_sys::Array::new();
    for warning in validation.warnings.iter() {
        warnings.push(&JsValue::from_str(warning));
    }
    js_sys::Reflect::set(&obj, &"warnings".into(), &warnings)?;

    Ok(obj.into())
}

/// Validate that results are within expected bounds (e.g., top_k).
///
/// # Arguments
/// * `results` - Array of `[id, score]` pairs
/// * `max_results` - Maximum expected number of results (None = no check)
///
/// # Returns
/// Validation result object with `is_valid` boolean and `warnings` array.
#[cfg(feature = "wasm")]
#[wasm_bindgen(js_name = validateBounds)]
pub fn validate_bounds(
    results: &JsValue,
    max_results: Option<usize>,
) -> Result<JsValue, JsValue> {
    let results_vec = js_to_results(results)?;
    let validation = crate::validate::validate_bounds(&results_vec, max_results);

    let obj = js_sys::Object::new();
    js_sys::Reflect::set(
        &obj,
        &"is_valid".into(),
        &JsValue::from_bool(validation.is_valid),
    )?;

    let warnings = js_sys::Array::new();
    for warning in validation.warnings.iter() {
        warnings.push(&JsValue::from_str(warning));
    }
    js_sys::Reflect::set(&obj, &"warnings".into(), &warnings)?;

    Ok(obj.into())
}
