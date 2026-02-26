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

// ── cellular_automaton ────────────────────────────────────────────────────────

fn bench_cellular_automaton(c: &mut Criterion) {
    let mut group = c.benchmark_group("cellular_automaton");
    for &size in &[256usize, 512, 1024] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::cellular_automaton(size, size, 0.45, 5, 5, 4, Some(42)));
        });
    }
    group.finish()
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

// ── reaction_diffusion ────────────────────────────────────────────────────────
//
// Each iteration touches every cell; reduce sample size for larger grids.

fn bench_reaction_diffusion(c: &mut Criterion) {
    let mut group = c.benchmark_group("reaction_diffusion");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::reaction_diffusion(size, size, 500, 0.055, 0.062, Some(42)));
        });
    }
    group.finish();
}

// ── eden_growth ───────────────────────────────────────────────────────────────

fn bench_eden_growth(c: &mut Criterion) {
    let mut group = c.benchmark_group("eden_growth");
    for &n in &[1_000usize, 5_000, 10_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| nlmrs::eden_growth(256, 256, n, Some(42)));
        });
    }
    group.finish();
}

// ── fractal_brownian_surface ──────────────────────────────────────────────────

fn bench_fractal_brownian_surface(c: &mut Criterion) {
    let mut group = c.benchmark_group("fractal_brownian_surface");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::fractal_brownian_surface(size, size, 0.5, Some(42)));
        });
    }
    group.finish();
}

// ── landscape_gradient ────────────────────────────────────────────────────────

fn bench_landscape_gradient(c: &mut Criterion) {
    let mut group = c.benchmark_group("landscape_gradient");
    for &size in &[512usize, 1024, 2048] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::landscape_gradient(size, size, Some(45.0), 2.0, Some(42)));
        });
    }
    group.finish();
}

// ── diffusion_limited_aggregation ─────────────────────────────────────────────
//
// Sequential random walk — reduce sample size and extend measurement time.

fn bench_diffusion_limited_aggregation(c: &mut Criterion) {
    let mut group = c.benchmark_group("diffusion_limited_aggregation");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &n in &[500usize, 1_000, 2_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| nlmrs::diffusion_limited_aggregation(128, 128, n, Some(42)));
        });
    }
    group.finish();
}

// ── simplex_noise ─────────────────────────────────────────────────────────────

fn bench_simplex_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("simplex_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::simplex_noise(size, size, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── invasion_percolation ──────────────────────────────────────────────────────
//
// Sequential BFS/heap traversal — reduce sample size for large n.

fn bench_invasion_percolation(c: &mut Criterion) {
    let mut group = c.benchmark_group("invasion_percolation");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &n in &[500usize, 2_000, 5_000] {
        group.bench_with_input(BenchmarkId::from_parameter(n), &n, |b, &n| {
            b.iter(|| nlmrs::invasion_percolation(128, 128, n, Some(42)));
        });
    }
    group.finish();
}

// ── gaussian_blobs ────────────────────────────────────────────────────────────

fn bench_gaussian_blobs(c: &mut Criterion) {
    let mut group = c.benchmark_group("gaussian_blobs");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::gaussian_blobs(size, size, 50, 5.0, Some(42)));
        });
    }
    group.finish();
}

// ── ising_model ───────────────────────────────────────────────────────────────
//
// Sequential random-sequential updates — reduce sample size for large grids.

fn bench_ising_model(c: &mut Criterion) {
    let mut group = c.benchmark_group("ising_model");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::ising_model(size, size, 0.4, 500, Some(42)));
        });
    }
    group.finish();
}

// ── voronoi_distance ──────────────────────────────────────────────────────────

fn bench_voronoi_distance(c: &mut Criterion) {
    let mut group = c.benchmark_group("voronoi_distance");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::voronoi_distance(size, size, 50, Some(42)));
        });
    }
    group.finish();
}

// ── sine_composite ────────────────────────────────────────────────────────────

fn bench_sine_composite(c: &mut Criterion) {
    let mut group = c.benchmark_group("sine_composite");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::sine_composite(size, size, 8, Some(42)));
        });
    }
    group.finish();
}

// ── curl_noise ────────────────────────────────────────────────────────────────

fn bench_curl_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("curl_noise");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::curl_noise(size, size, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── hydraulic_erosion ─────────────────────────────────────────────────────────

fn bench_hydraulic_erosion(c: &mut Criterion) {
    let mut group = c.benchmark_group("hydraulic_erosion");
    group.sample_size(20);
    group.measurement_time(Duration::from_secs(15));
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::hydraulic_erosion(size, size, 500, Some(42)));
        });
    }
    group.finish();
}

// ── levy_flight ───────────────────────────────────────────────────────────────

fn bench_levy_flight(c: &mut Criterion) {
    let mut group = c.benchmark_group("levy_flight");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::levy_flight(size, size, 1000, Some(42)));
        });
    }
    group.finish();
}

// ── poisson_disk ──────────────────────────────────────────────────────────────

fn bench_poisson_disk(c: &mut Criterion) {
    let mut group = c.benchmark_group("poisson_disk");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::poisson_disk(size, size, 5.0, Some(42)));
        });
    }
    group.finish();
}

// ── gabor_noise ───────────────────────────────────────────────────────────────

fn bench_gabor_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("gabor_noise");
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::gabor_noise(size, size, 4.0, 500, Some(42)));
        });
    }
    group.finish();
}

// ── spot_noise ────────────────────────────────────────────────────────────────

fn bench_spot_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("spot_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::spot_noise(size, size, 200, Some(42)));
        });
    }
    group.finish();
}

// ── anisotropic_noise ─────────────────────────────────────────────────────────

fn bench_anisotropic_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("anisotropic_noise");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::anisotropic_noise(size, size, 4.0, 6, 45.0, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── tiled_noise ───────────────────────────────────────────────────────────────

fn bench_tiled_noise(c: &mut Criterion) {
    let mut group = c.benchmark_group("tiled_noise");
    for &size in &[128usize, 256, 512, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::tiled_noise(size, size, 4.0, Some(42)));
        });
    }
    group.finish();
}

// ── brownian_motion ───────────────────────────────────────────────────────────

fn bench_brownian_motion(c: &mut Criterion) {
    let mut group = c.benchmark_group("brownian_motion");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::brownian_motion(size, size, 5000, Some(42)));
        });
    }
    group.finish();
}

// ── forest_fire ───────────────────────────────────────────────────────────────

fn bench_forest_fire(c: &mut Criterion) {
    let mut group = c.benchmark_group("forest_fire");
    group.sample_size(20).measurement_time(Duration::from_secs(15));
    for &size in &[64usize, 128, 256] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::forest_fire(size, size, 0.02, 0.001, 500, Some(42)));
        });
    }
    group.finish();
}

// ── river_network ─────────────────────────────────────────────────────────────

fn bench_river_network(c: &mut Criterion) {
    let mut group = c.benchmark_group("river_network");
    for &size in &[128usize, 256, 512] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::river_network(size, size, Some(42)));
        });
    }
    group.finish();
}

// ── hexagonal_voronoi ─────────────────────────────────────────────────────────

fn bench_hexagonal_voronoi(c: &mut Criterion) {
    let mut group = c.benchmark_group("hexagonal_voronoi");
    for &size in &[128usize, 256, 512, 1000] {
        group.bench_with_input(BenchmarkId::from_parameter(size), &size, |b, &size| {
            b.iter(|| nlmrs::hexagonal_voronoi(size, size, 50, Some(42)));
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
    bench_hybrid_noise,
    bench_value_noise,
    bench_turbulence,
    bench_domain_warp,
    bench_mosaic,
    bench_rectangular_cluster,
    bench_percolation,
    bench_binary_space_partitioning,
    bench_cellular_automaton,
    bench_neighbourhood_clustering,
    bench_spectral_synthesis,
    bench_diffusion_limited_aggregation,
    bench_reaction_diffusion,
    bench_eden_growth,
    bench_fractal_brownian_surface,
    bench_landscape_gradient,
    bench_simplex_noise,
    bench_invasion_percolation,
    bench_gaussian_blobs,
    bench_ising_model,
    bench_voronoi_distance,
    bench_sine_composite,
    bench_curl_noise,
    bench_hydraulic_erosion,
    bench_levy_flight,
    bench_poisson_disk,
    bench_gabor_noise,
    bench_spot_noise,
    bench_anisotropic_noise,
    bench_tiled_noise,
    bench_brownian_motion,
    bench_forest_fire,
    bench_river_network,
    bench_hexagonal_voronoi,
);
criterion_main!(benches);
