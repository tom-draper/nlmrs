use std::time::Duration;

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

// ── midpoint_displacement ─────────────────────────────────────────────────────

fn bench_midpoint_displacement(c: &mut Criterion) {
    let mut group = c.benchmark_group("midpoint_displacement");
    for &size in &[128usize, 256, 512, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::midpoint_displacement(size, size, 0.8, Some(42)));
        });
    }
    group.finish();
}

// ── fbm_noise ─────────────────────────────────────────────────────────────────

fn bench_fbm_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("fbm_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::fbm_noise(size, size, 4.0, 6, 0.5, 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── distance_gradient (EDT) ───────────────────────────────────────────────────

fn bench_distance_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("distance_gradient");
    for &size in &[256usize, 512, 1024, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::distance_gradient(size, size, Some(42)));
        });
    }
    group.finish();
}

// ── hill_grow ─────────────────────────────────────────────────────────────────
//
// Reduced sample size — each iteration involves many weighted-sampled kernel
// applications and can run for several hundred milliseconds at larger n.

fn bench_hill_grow(c: &mut Criterion) {
    let mut group = c.benchmark_group("hill_grow");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &n in &[1_000usize, 5_000, 10_000, 50_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| nlmrs::hill_grow(128, 128, n, true, None, false, Some(42)));
        });
    }
    group.finish();
}

// ── random_element ────────────────────────────────────────────────────────────

fn bench_random_element(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_element");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::random_element(size, size, 50_000.0, Some(42)));
        });
    }
    group.finish();
}

// ── classify / threshold (operations) ────────────────────────────────────────

fn bench_classify(c: &mut Criterion) {
    let mut group = c.benchmark_group("classify");
    for &size in &[512usize, 1024, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let grid = nlmrs::random(size, size, Some(1));
            b.iter(|| {
                let mut g = grid.clone();
                nlmrs::classify(&mut g, 5);
            });
        });
    }
    group.finish();
}

fn bench_threshold(c: &mut Criterion) {
    let mut group = c.benchmark_group("threshold");
    for &size in &[512usize, 1024, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            let grid = nlmrs::random(size, size, Some(1));
            b.iter(|| {
                let mut g = grid.clone();
                nlmrs::threshold(&mut g, 0.5);
            });
        });
    }
    group.finish();
}

// ── ridged_noise ──────────────────────────────────────────────────────────────

fn bench_ridged_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("ridged_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::ridged_noise(size, size, 4.0, 6, 0.5, 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── billow_noise ──────────────────────────────────────────────────────────────

fn bench_billow_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("billow_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::billow_noise(size, size, 4.0, 6, 0.5, 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── worley_noise ──────────────────────────────────────────────────────────────

fn bench_worley_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("worley_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::worley_noise(size, size, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── gaussian_field ────────────────────────────────────────────────────────────

fn bench_gaussian_field(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_field");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::gaussian_field(size, size, 20.0, Some(42)));
        });
    }
    group.finish();
}

// ── random_cluster ────────────────────────────────────────────────────────────

fn bench_random_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("random_cluster");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::random_cluster(size, size, 200, Some(42)));
        });
    }
    group.finish();
}

// ── hybrid_noise ──────────────────────────────────────────────────────────────

fn bench_hybrid_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("hybrid_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::hybrid_noise(size, size, 4.0, 6, 0.5, 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── value_noise ───────────────────────────────────────────────────────────────

fn bench_value_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("value_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::value_noise(size, size, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── turbulence ────────────────────────────────────────────────────────────────

fn bench_turbulence(c: &mut Criterion) {
    let mut group = c.benchmark_group("turbulence");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::turbulence(size, size, 4.0, 6, 0.5, 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── domain_warp ───────────────────────────────────────────────────────────────

fn bench_domain_warp(c: &mut Criterion) {
    let mut group = c.benchmark_group("domain_warp");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::domain_warp(size, size, 4.0, 1.0, Some(42)));
        });
    }
    group.finish();
}

// ── mosaic ────────────────────────────────────────────────────────────────────

fn bench_mosaic(c: &mut Criterion) {
    let mut group = c.benchmark_group("mosaic");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::mosaic(size, size, 200, Some(42)));
        });
    }
    group.finish();
}

// ── rectangular_cluster ───────────────────────────────────────────────────────

fn bench_rectangular_cluster(c: &mut Criterion) {
    let mut group = c.benchmark_group("rectangular_cluster");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::rectangular_cluster(size, size, 200, Some(42)));
        });
    }
    group.finish();
}

// ── percolation ───────────────────────────────────────────────────────────────

fn bench_percolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("percolation");
    for &size in &[256usize, 512, 1024, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::percolation(size, size, 0.5, Some(42)));
        });
    }
    group.finish();
}

// ── binary_space_partitioning ─────────────────────────────────────────────────

fn bench_binary_space_partitioning(c: &mut Criterion) {
    let mut group = c.benchmark_group("binary_space_partitioning");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::binary_space_partitioning(size, size, 100, Some(42)));
        });
    }
    group.finish();
}

// ── neighbourhood_clustering ──────────────────────────────────────────────────

fn bench_neighbourhood_clustering(c: &mut Criterion) {
    let mut group = c.benchmark_group("neighbourhood_clustering");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::neighbourhood_clustering(size, size, 5, 10, Some(42)));
        });
    }
    group.finish()
}

// ── spectral_synthesis ────────────────────────────────────────────────────────

fn bench_spectral_synthesis(c: &mut Criterion) {
    let mut group = c.benchmark_group("spectral_synthesis");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::spectral_synthesis(size, size, 2.0, Some(42)));
        });
    }
    group.finish()
}

criterion_group!(
    benches,
    bench_midpoint_displacement,
    bench_fbm_noise,
    bench_distance_gradient,
    bench_hill_grow,
    bench_random_element,
    bench_classify,
    bench_threshold,
    bench_ridged_noise,
    bench_billow_noise,
    bench_worley_noise,
    bench_gaussian_field,
    bench_random_cluster,
    bench_hybrid_noise,
    bench_value_noise,
    bench_turbulence,
    bench_domain_warp,
    bench_mosaic,
    bench_rectangular_cluster,
    bench_percolation,
    bench_binary_space_partitioning,
    bench_neighbourhood_clustering,
    bench_spectral_synthesis,
);
criterion_main!(benches);
