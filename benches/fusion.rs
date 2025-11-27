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
    use rank_fusion::{borda_multi, combmnz_multi, combsum_multi, rrf_multi, FusionConfig};

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

    g.finish();
}

criterion_group!(benches, bench_algorithms, bench_multi);
criterion_main!(benches);
