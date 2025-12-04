//! Comprehensive benchmarks for all rank-fusion algorithms.
//!
//! Run: `cargo bench --bench comprehensive`

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use rank_fusion::*;
use std::collections::HashMap;

fn generate_test_data(n: usize) -> (Vec<(String, f32)>, Vec<(String, f32)>) {
    let mut list1 = Vec::new();
    let mut list2 = Vec::new();
    
    for i in 0..n {
        list1.push((format!("doc_{}", i), (n - i) as f32));
        if i % 2 == 0 {
            list2.push((format!("doc_{}", i), (n - i) as f32 * 0.9));
        }
    }
    
    (list1, list2)
}

fn generate_multi_list_data(n_lists: usize, items_per_list: usize) -> Vec<Vec<(String, f32)>> {
    (0..n_lists)
        .map(|list_idx| {
            (0..items_per_list)
                .map(|i| {
                    let doc_id = format!("doc_{}_{}", list_idx, i);
                    let score = (items_per_list - i) as f32 * (1.0 - list_idx as f32 * 0.1);
                    (doc_id, score)
                })
                .collect()
        })
        .collect()
}

fn bench_rank_based_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("rank_based_fusion");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let (list1, list2) = generate_test_data(*size);
        
        group.bench_with_input(BenchmarkId::new("rrf", size), size, |b, _| {
            b.iter(|| rrf(black_box(&list1), black_box(&list2)))
        });
        
        group.bench_with_input(BenchmarkId::new("isr", size), size, |b, _| {
            b.iter(|| isr(black_box(&list1), black_box(&list2)))
        });
        
        group.bench_with_input(BenchmarkId::new("borda", size), size, |b, _| {
            b.iter(|| borda(black_box(&list1), black_box(&list2)))
        });
    }
    
    group.finish();
}

fn bench_score_based_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("score_based_fusion");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let (list1, list2) = generate_test_data(*size);
        
        group.bench_with_input(BenchmarkId::new("combsum", size), size, |b, _| {
            b.iter(|| combsum(black_box(&list1), black_box(&list2)))
        });
        
        group.bench_with_input(BenchmarkId::new("combmnz", size), size, |b, _| {
            b.iter(|| combmnz(black_box(&list1), black_box(&list2)))
        });
        
        group.bench_with_input(BenchmarkId::new("dbsf", size), size, |b, _| {
            b.iter(|| dbsf(black_box(&list1), black_box(&list2)))
        });
        
        group.bench_with_input(BenchmarkId::new("standardized", size), size, |b, _| {
            b.iter(|| standardized(black_box(&list1), black_box(&list2)))
        });
    }
    
    group.finish();
}

fn bench_weighted_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("weighted_fusion");
    
    for size in [10, 50, 100, 500, 1000].iter() {
        let (list1, list2) = generate_test_data(*size);
        let config = WeightedConfig::new(0.3, 0.7).with_normalize(true);
        
        group.bench_with_input(BenchmarkId::new("weighted", size), size, |b, _| {
            b.iter(|| weighted(black_box(&list1), black_box(&list2), black_box(config)))
        });
    }
    
    group.finish();
}

fn bench_multi_list_fusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("multi_list_fusion");
    
    for (n_lists, items_per_list) in [(2, 50), (3, 50), (5, 50), (10, 50)].iter() {
        let lists = generate_multi_list_data(*n_lists, *items_per_list);
        let slices: Vec<&[(String, f32)]> = lists.iter().map(|l| l.as_slice()).collect();
        
        group.bench_with_input(
            BenchmarkId::new("rrf_multi", format!("{}x{}", n_lists, items_per_list)),
            &slices,
            |b, slices| {
                b.iter(|| rrf_multi(black_box(slices), black_box(RrfConfig::default())))
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("combsum_multi", format!("{}x{}", n_lists, items_per_list)),
            &slices,
            |b, slices| {
                b.iter(|| combsum_multi(black_box(slices), black_box(FusionConfig::default())))
            },
        );
    }
    
    group.finish();
}

fn bench_explainability(c: &mut Criterion) {
    let mut group = c.benchmark_group("explainability");
    
    for size in [10, 50, 100].iter() {
        let lists = generate_multi_list_data(3, *size);
        let slices: Vec<&[(String, f32)]> = lists.iter().map(|l| l.as_slice()).collect();
        let retriever_ids = vec![
            RetrieverId::new("bm25"),
            RetrieverId::new("dense"),
            RetrieverId::new("sparse"),
        ];
        
        group.bench_with_input(BenchmarkId::new("rrf_explain", size), size, |b, _| {
            b.iter(|| {
                rrf_explain(
                    black_box(&slices),
                    black_box(&retriever_ids),
                    black_box(RrfConfig::default()),
                )
            })
        });
    }
    
    group.finish();
}

fn bench_configuration_overhead(c: &mut Criterion) {
    let mut group = c.benchmark_group("configuration_overhead");
    
    let (list1, list2) = generate_test_data(100);
    
    group.bench_function("rrf_default", |b| {
        b.iter(|| rrf(black_box(&list1), black_box(&list2)))
    });
    
    group.bench_function("rrf_with_config", |b| {
        let config = RrfConfig::default().with_top_k(10);
        b.iter(|| rrf_with_config(black_box(&list1), black_box(&list2), black_box(config)))
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_rank_based_fusion,
    bench_score_based_fusion,
    bench_weighted_fusion,
    bench_multi_list_fusion,
    bench_explainability,
    bench_configuration_overhead
);
criterion_main!(benches);

