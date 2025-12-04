//! Rank fusion evaluation library.
//!
//! Provides evaluation infrastructure for rank fusion methods on both
//! synthetic scenarios and real-world IR datasets.

pub mod datasets;
pub mod metrics;
pub mod real_world;
pub mod dataset_loaders;
pub mod dataset_converters;
pub mod dataset_registry;
pub mod dataset_validator;
pub mod dataset_statistics;
pub mod evaluate_real_world;

#[cfg(test)]
mod integration_tests;

