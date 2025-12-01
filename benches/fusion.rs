use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rank_fusion::{borda, combmnz, combsum, rrf, rrf_with_config, RrfConfig};

fn ranked(n: usize, prefix: &str) -> Vec<(String, f32)> {
    (0..n)
        .map(|i| (format!("{prefix}_{i}"), 1.0 - i as f32 / n as f32))
        .collect()
}

fn with_overlap(mut results: Vec<(String, f32)>, n: usize, frac: f32) -> Vec<(String, f32)> {
    let count = (n as f32 * frac) as usize;
    for i in 0..count.min(results.len()) {
        results[i].0 = format!("shared_{i}");
    }
    results
}

fn bench_algorithms(c: &mut Criterion) {
    let mut g = c.benchmark_group("fusion");

    for &n in &[100, 1000] {
        let a = ranked(n, "a");
        let b = with_overlap(ranked(n, "b"), n, 0.3);

        g.bench_with_input(BenchmarkId::new("rrf", n), &n, |bench, _| {
            bench.iter(|| black_box(rrf(&a, &b)));
        });

        g.bench_with_input(BenchmarkId::new("rrf_k20", n), &n, |bench, _| {
            bench.iter(|| black_box(rrf_with_config(&a, &b, RrfConfig::new(20))));
        });

        g.bench_with_input(BenchmarkId::new("combsum", n), &n, |bench, _| {
            bench.iter(|| black_box(combsum(&a, &b)));
        });

        g.bench_with_input(BenchmarkId::new("combmnz", n), &n, |bench, _| {
            bench.iter(|| black_box(combmnz(&a, &b)));
        });

        g.bench_with_input(BenchmarkId::new("borda", n), &n, |bench, _| {
            bench.iter(|| black_box(borda(&a, &b)));
        });
    }

    g.finish();
}

fn bench_multi(c: &mut Criterion) {
    use rank_fusion::{borda_multi, combmnz_multi, combsum_multi, dbsf_multi, isr_multi, rrf_multi, rrf_weighted, FusionConfig};

    let mut g = c.benchmark_group("multi");

    let lists: Vec<Vec<(String, f32)>> = (0..5).map(|i| ranked(100, &format!("list{i}"))).collect();
    let list_refs: Vec<&[(String, f32)]> = lists.iter().map(|v| v.as_slice()).collect();

    g.bench_function("rrf_multi_5x100", |bench| {
        bench.iter(|| black_box(rrf_multi(&list_refs, RrfConfig::default())));
    });

    g.bench_function("borda_multi_5x100", |bench| {
        bench.iter(|| black_box(borda_multi(&list_refs, FusionConfig::default())));
    });

    g.bench_function("combsum_multi_5x100", |bench| {
        bench.iter(|| black_box(combsum_multi(&list_refs, FusionConfig::default())));
    });

    g.bench_function("combmnz_multi_5x100", |bench| {
        bench.iter(|| black_box(combmnz_multi(&list_refs, FusionConfig::default())));
    });

    g.bench_function("dbsf_multi_5x100", |bench| {
        bench.iter(|| black_box(dbsf_multi(&list_refs, FusionConfig::default())));
    });

    g.bench_function("isr_multi_5x100", |bench| {
        bench.iter(|| black_box(isr_multi(&list_refs, RrfConfig::default())));
    });

    // Weighted RRF with 5 lists
    let weights = [0.2, 0.2, 0.2, 0.2, 0.2];
    g.bench_function("rrf_weighted_5x100", |bench| {
        bench.iter(|| black_box(rrf_weighted(&list_refs, &weights, RrfConfig::default()).unwrap()));
    });

    g.finish();
}

fn bench_edge_cases(c: &mut Criterion) {
    use rank_fusion::{rrf, rrf_multi, rrf_with_config, RrfConfig};

    let mut g = c.benchmark_group("edge_cases");

    // Empty lists
    let empty: Vec<(String, f32)> = Vec::new();
    let non_empty = ranked(100, "doc");
    g.bench_function("rrf_one_empty", |bench| {
        bench.iter(|| black_box(rrf(&empty, &non_empty)));
    });

    // Large overlap
    let a = ranked(100, "a");
    let b: Vec<(String, f32)> = a.iter().map(|(id, s)| (id.clone(), *s)).collect();
    g.bench_function("rrf_identical_lists", |bench| {
        bench.iter(|| black_box(rrf(&a, &b)));
    });

    // Many small lists
    let many_lists: Vec<Vec<(String, f32)>> = (0..20).map(|i| ranked(10, &format!("list{i}"))).collect();
    let many_refs: Vec<&[(String, f32)]> = many_lists.iter().map(|v| v.as_slice()).collect();
    g.bench_function("rrf_multi_20x10", |bench| {
        bench.iter(|| black_box(rrf_multi(&many_refs, RrfConfig::default())));
    });

    // High k value
    g.bench_function("rrf_k_1000", |bench| {
        bench.iter(|| black_box(rrf_with_config(&a, &b, RrfConfig::new(1000))));
    });

    g.finish();
}

criterion_group!(benches, bench_algorithms, bench_multi, bench_edge_cases);
criterion_main!(benches);
