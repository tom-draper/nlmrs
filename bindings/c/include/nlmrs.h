#ifndef NLMRS_H
#define NLMRS_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

// A 2-D grid returned by every NLM generator.
//
// `data` points to a heap-allocated row-major array of `rows * cols` doubles
// with values in [0, 1]. Call `nlmrs_free` exactly once when you are done.
typedef struct NlmGrid {
    // Heap-allocated row-major data, length = rows * cols.
    double *data;
    uintptr_t rows;
    uintptr_t cols;
} NlmGrid;

// Free a grid returned by any `nlmrs_*` function.
//
// Must be called exactly once per grid. Behaviour is undefined if called on a
// zeroed struct or a grid that has already been freed.
void nlmrs_free(struct NlmGrid grid);

// Spatially random NLM. Values in [0, 1).
//
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_random(uintptr_t rows, uintptr_t cols, const uint64_t *seed);

// Random element nearest-neighbour NLM. Values in [0, 1).
//
// @param n     Number of labelled seed elements to place.
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_random_element(uintptr_t rows, uintptr_t cols, double n, const uint64_t *seed);

// Linear planar gradient. Values in [0, 1).
//
// @param direction  Pointer to gradient direction in degrees [0, 360), or NULL for random.
// @param seed       Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_planar_gradient(uintptr_t rows,
                                     uintptr_t cols,
                                     const double *direction,
                                     const uint64_t *seed);

// Symmetric edge gradient — zero at both edges, peak in the middle. Values in [0, 1).
//
// @param direction  Pointer to gradient direction in degrees [0, 360), or NULL for random.
// @param seed       Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_edge_gradient(uintptr_t rows,
                                   uintptr_t cols,
                                   const double *direction,
                                   const uint64_t *seed);

// Radial distance gradient from a random centre point. Values in [0, 1).
//
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_distance_gradient(uintptr_t rows, uintptr_t cols, const uint64_t *seed);

// Sinusoidal wave gradient. Values in [0, 1).
//
// @param period     Wave period — smaller values produce larger waves.
// @param direction  Pointer to wave direction in degrees [0, 360), or NULL for random.
// @param seed       Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_wave_gradient(uintptr_t rows,
                                   uintptr_t cols,
                                   double period,
                                   const double *direction,
                                   const uint64_t *seed);

// Diamond-square (midpoint displacement) fractal terrain. Values in [0, 1).
//
// @param h     Spatial autocorrelation — 0 = rough, 1 = smooth.
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_midpoint_displacement(uintptr_t rows,
                                           uintptr_t cols,
                                           double h,
                                           const uint64_t *seed);

// Hill-grow NLM. Values in [0, 1).
//
// @param n            Number of iterations.
// @param runaway      If non-zero, hills cluster via weighted sampling.
// @param kernel_data  Flat row-major convolution kernel, or NULL for the default 3×3 diamond.
// @param kernel_size  Side length of the square kernel (ignored when kernel_data is NULL).
// @param only_grow    If non-zero the surface only accumulates, never shrinks.
// @param seed         Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_hill_grow(uintptr_t rows,
                               uintptr_t cols,
                               uintptr_t n,
                               bool runaway,
                               const double *kernel_data,
                               uintptr_t kernel_size,
                               bool only_grow,
                               const uint64_t *seed);

// Single-layer Perlin noise. Values in [0, 1).
//
// @param scale  Noise frequency — higher values produce more features per unit area.
// @param seed   Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_perlin_noise(uintptr_t rows,
                                  uintptr_t cols,
                                  double scale,
                                  const uint64_t *seed);

// Fractal Brownian motion — layered Perlin noise. Values in [0, 1).
//
// @param scale       Base noise frequency.
// @param octaves     Number of noise layers to combine.
// @param persistence Amplitude scaling per octave.
// @param lacunarity  Frequency scaling per octave.
// @param seed        Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_fbm_noise(uintptr_t rows,
                               uintptr_t cols,
                               double scale,
                               uintptr_t octaves,
                               double persistence,
                               double lacunarity,
                               const uint64_t *seed);

// Ridged multifractal noise. Values in [0, 1).
//
// @param scale       Base noise frequency.
// @param octaves     Number of noise layers to combine.
// @param persistence Amplitude scaling per octave.
// @param lacunarity  Frequency scaling per octave.
// @param seed        Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_ridged_noise(uintptr_t rows,
                                  uintptr_t cols,
                                  double scale,
                                  uintptr_t octaves,
                                  double persistence,
                                  double lacunarity,
                                  const uint64_t *seed);

// Billow noise — rounded cloud- and hill-like patterns. Values in [0, 1).
//
// @param scale       Base noise frequency.
// @param octaves     Number of noise layers to combine.
// @param persistence Amplitude scaling per octave.
// @param lacunarity  Frequency scaling per octave.
// @param seed        Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_billow_noise(uintptr_t rows,
                                  uintptr_t cols,
                                  double scale,
                                  uintptr_t octaves,
                                  double persistence,
                                  double lacunarity,
                                  const uint64_t *seed);

// Worley (cellular) noise — territory / patch patterns. Values in [0, 1).
//
// @param scale  Seed-point frequency; higher values produce smaller cells.
// @param seed   Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_worley_noise(uintptr_t rows,
                                  uintptr_t cols,
                                  double scale,
                                  const uint64_t *seed);

// Gaussian random field — spatially correlated noise. Values in [0, 1).
//
// @param sigma  Gaussian kernel standard deviation in cells (controls correlation length).
// @param seed   Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_gaussian_field(uintptr_t rows,
                                    uintptr_t cols,
                                    double sigma,
                                    const uint64_t *seed);

// Random cluster NLM via fault-line cuts. Values in [0, 1).
//
// @param n     Number of fault-line cuts. Higher values produce finer-grained clusters.
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_random_cluster(uintptr_t rows,
                                    uintptr_t cols,
                                    uintptr_t n,
                                    const uint64_t *seed);

// Hybrid multifractal noise. Values in [0, 1).
//
// @param scale       Base noise frequency.
// @param octaves     Number of noise layers to combine.
// @param persistence Amplitude scaling per octave.
// @param lacunarity  Frequency scaling per octave.
// @param seed        Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_hybrid_noise(uintptr_t rows,
                                  uintptr_t cols,
                                  double scale,
                                  uintptr_t octaves,
                                  double persistence,
                                  double lacunarity,
                                  const uint64_t *seed);

// Value noise — interpolated lattice noise. Values in [0, 1).
//
// @param scale  Noise frequency — higher values produce more features per unit area.
// @param seed   Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_value_noise(uintptr_t rows,
                                 uintptr_t cols,
                                 double scale,
                                 const uint64_t *seed);

// Turbulence — fBm with absolute-value fold per octave. Values in [0, 1).
//
// @param scale       Base noise frequency.
// @param octaves     Number of noise layers to combine.
// @param persistence Amplitude scaling per octave.
// @param lacunarity  Frequency scaling per octave.
// @param seed        Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_turbulence(uintptr_t rows,
                                uintptr_t cols,
                                double scale,
                                uintptr_t octaves,
                                double persistence,
                                double lacunarity,
                                const uint64_t *seed);

// Domain-warped Perlin noise — organic, swirling patterns. Values in [0, 1).
//
// @param scale         Coordinate frequency.
// @param warp_strength Displacement magnitude applied to sample coordinates.
// @param seed          Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_domain_warp(uintptr_t rows,
                                 uintptr_t cols,
                                 double scale,
                                 double warp_strength,
                                 const uint64_t *seed);

// Mosaic NLM — discrete Voronoi patch map with flat-coloured regions. Values in [0, 1).
//
// @param n     Number of Voronoi seed points to place.
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_mosaic(uintptr_t rows, uintptr_t cols, uintptr_t n, const uint64_t *seed);

// Rectangular cluster NLM — overlapping random axis-aligned rectangles. Values in [0, 1).
//
// @param n     Number of rectangles to place.
// @param seed  Pointer to a u64 seed, or NULL for a random seed.
struct NlmGrid nlmrs_rectangular_cluster(uintptr_t rows,
                                         uintptr_t cols,
                                         uintptr_t n,
                                         const uint64_t *seed);

#endif /* NLMRS_H */
