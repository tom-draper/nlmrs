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
);
criterion_main!(benches);
