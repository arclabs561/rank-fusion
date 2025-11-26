//! Benchmarks for rank fusion algorithms.
//!
//! Run with: `cargo bench`
//!
//! Expected results (M1 Mac):
//! - RRF @ 100 results: ~13 µs, 7.8 Melem/s
//! - RRF preallocated: ~9.6 µs, 10.4 Melem/s (30% faster)
//! - CombMNZ: ~13.5 µs, 7.4 Melem/s

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use rerank::{
    fuse_borda, fuse_combmnz, fuse_combsum, fuse_rrf, fuse_rrf_into, fuse_rrf_multi,
    fuse_weighted, RrfConfig, WeightedConfig,
};
use std::hint::black_box;

fn generate_results(n: usize, prefix: &str, overlap_fraction: f32) -> Vec<(String, f32)> {
    let mut results = Vec::with_capacity(n);
    for i in 0..n {
        let id = format!("{}_{}", prefix, i);
        let score = 1.0 - (i as f32 / n as f32);
        results.push((id, score));
    }

    // Add overlapping IDs
    let overlap_count = (n as f32 * overlap_fraction) as usize;
    for i in 0..overlap_count {
        if i < n {
            results[i].0 = format!("shared_{}", i);
        }
    }

    results
}

fn bench_fusion_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("fusion");

    for n in [10, 100, 1000].iter() {
        let results_a = generate_results(*n, "dense", 0.3);
        let results_b = generate_results(*n, "sparse", 0.3);

        group.throughput(Throughput::Elements(*n as u64));

        // RRF (k=60)
        group.bench_with_input(BenchmarkId::new("rrf", n), n, |b, _| {
            b.iter(|| black_box(fuse_rrf(results_a.clone(), results_b.clone(), RrfConfig::default())))
        });

        // RRF preallocated
        group.bench_with_input(BenchmarkId::new("rrf_preallocated", n), n, |b, _| {
            let mut output = Vec::with_capacity(*n * 2);
            b.iter(|| {
                fuse_rrf_into(&results_a, &results_b, RrfConfig::default(), &mut output);
                black_box(output.len())
            })
        });

        // Weighted (equal)
        group.bench_with_input(BenchmarkId::new("weighted", n), n, |b, _| {
            b.iter(|| black_box(fuse_weighted(&results_a, &results_b, WeightedConfig::default())))
        });

        // CombSUM
        group.bench_with_input(BenchmarkId::new("combsum", n), n, |b, _| {
            b.iter(|| black_box(fuse_combsum(&results_a, &results_b)))
        });

        // CombMNZ
        group.bench_with_input(BenchmarkId::new("combmnz", n), n, |b, _| {
            b.iter(|| black_box(fuse_combmnz(&results_a, &results_b)))
        });

        // Borda
        group.bench_with_input(BenchmarkId::new("borda", n), n, |b, _| {
            b.iter(|| black_box(fuse_borda(&results_a, &results_b)))
        });
    }

    group.finish();
}

fn bench_multi_source(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_source");

    let n = 100;
    let lists: Vec<Vec<(String, f32)>> = vec![
        generate_results(n, "dense", 0.3),
        generate_results(n, "sparse", 0.3),
        generate_results(n, "kg", 0.3),
    ];

    group.throughput(Throughput::Elements((n * 3) as u64));

    group.bench_function("rrf_3way", |b| {
        b.iter(|| black_box(fuse_rrf_multi(&lists, RrfConfig::default())))
    });

    group.finish();
}

fn bench_overlap_impact(c: &mut Criterion) {
    let mut group = c.benchmark_group("overlap");

    let n = 100;

    for overlap in [0.0, 0.5, 1.0].iter() {
        let results_a = generate_results(n, "dense", *overlap);
        let results_b = generate_results(n, "sparse", *overlap);

        let label = format!("{}%", (*overlap * 100.0) as u32);

        group.bench_with_input(BenchmarkId::new("rrf", &label), &overlap, |b, _| {
            b.iter(|| black_box(fuse_rrf(results_a.clone(), results_b.clone(), RrfConfig::default())))
        });

        group.bench_with_input(BenchmarkId::new("combmnz", &label), &overlap, |b, _| {
            b.iter(|| black_box(fuse_combmnz(&results_a, &results_b)))
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_fusion_strategies,
    bench_multi_source,
    bench_overlap_impact,
);
criterion_main!(benches);

