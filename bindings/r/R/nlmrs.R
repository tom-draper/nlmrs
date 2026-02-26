# User-facing nlm_* wrappers with roxygen2 documentation.
#
# These coerce dimensions to integer (R passes numerics by default) and
# forward to the thin .Call stubs in extendr-wrappers.R.

# ── Random ────────────────────────────────────────────────────────────────────

#' Spatially random NLM
#'
#' Each cell is drawn independently from a uniform distribution.
#'
#' @param rows Number of rows (positive integer).
#' @param cols Number of columns (positive integer).
#' @param seed Integer seed for reproducibility. \code{NULL} for random output.
#'
#' @return A numeric matrix of dimensions \code{rows × cols} with values in
#'   \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_random(10, 10, seed = 1L)
#' stopifnot(is.matrix(m), nrow(m) == 10, ncol(m) == 10)
nlm_random <- function(rows, cols, seed = NULL) {
  r_random(as.integer(rows), as.integer(cols),
           if (is.null(seed)) NULL else as.double(seed))
}

# ── Random element ────────────────────────────────────────────────────────────

#' Random element nearest-neighbour NLM
#'
#' Places \code{n} seed elements at random positions then fills the remainder
#' using nearest-neighbour interpolation.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of labelled seed elements (default 50 000).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_random_element(20, 20, n = 10, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_random_element <- function(rows, cols, n = 50000, seed = NULL) {
  r_random_element(as.integer(rows), as.integer(cols),
                   as.double(n),
                   if (is.null(seed)) NULL else as.double(seed))
}

# ── Planar gradient ───────────────────────────────────────────────────────────

#' Planar gradient NLM
#'
#' A linear gradient that falls across the grid in a given compass direction.
#'
#' @param rows      Number of rows.
#' @param cols      Number of columns.
#' @param direction Gradient direction in degrees \eqn{[0, 360)}. \code{NULL}
#'   picks a random direction.
#' @param seed      Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_planar_gradient(20, 20, direction = 90, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_planar_gradient <- function(rows, cols, direction = NULL, seed = NULL) {
  r_planar_gradient(as.integer(rows), as.integer(cols),
                    if (is.null(direction)) NULL else as.double(direction),
                    if (is.null(seed)) NULL else as.double(seed))
}

# ── Edge gradient ─────────────────────────────────────────────────────────────

#' Edge gradient NLM
#'
#' Symmetric gradient: zero at both edges, one at the midpoint between them.
#'
#' @param rows      Number of rows.
#' @param cols      Number of columns.
#' @param direction Gradient direction in degrees \eqn{[0, 360)}. \code{NULL}
#'   picks a random direction.
#' @param seed      Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_edge_gradient(20, 20, direction = 0, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_edge_gradient <- function(rows, cols, direction = NULL, seed = NULL) {
  r_edge_gradient(as.integer(rows), as.integer(cols),
                  if (is.null(direction)) NULL else as.double(direction),
                  if (is.null(seed)) NULL else as.double(seed))
}

# ── Distance gradient ─────────────────────────────────────────────────────────

#' Distance gradient NLM
#'
#' Radial gradient emanating outward from a single random origin point.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_distance_gradient(20, 20, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_distance_gradient <- function(rows, cols, seed = NULL) {
  r_distance_gradient(as.integer(rows), as.integer(cols),
                      if (is.null(seed)) NULL else as.double(seed))
}

# ── Wave gradient ─────────────────────────────────────────────────────────────

#' Wave gradient NLM
#'
#' Sinusoidal wave cycling \eqn{0 \to 1 \to 0} across the grid.
#'
#' @param rows      Number of rows.
#' @param cols      Number of columns.
#' @param period    Wave period — smaller values produce larger waves
#'   (default 2.5).
#' @param direction Wave direction in degrees \eqn{[0, 360)}. \code{NULL}
#'   picks a random direction.
#' @param seed      Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_wave_gradient(20, 20, period = 3, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_wave_gradient <- function(rows, cols, period = 2.5, direction = NULL, seed = NULL) {
  r_wave_gradient(as.integer(rows), as.integer(cols),
                  as.double(period),
                  if (is.null(direction)) NULL else as.double(direction),
                  if (is.null(seed)) NULL else as.double(seed))
}

# ── Midpoint displacement ─────────────────────────────────────────────────────

#' Midpoint displacement NLM
#'
#' Diamond-square fractal terrain. Produces spatially autocorrelated surfaces
#' resembling natural landscapes.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param h    Hurst exponent controlling spatial autocorrelation.
#'   \code{0} = very rough, \code{1} = very smooth (default 1.0).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_midpoint_displacement(200, 200, h = 0.8, seed = 42L)
#' stopifnot(is.matrix(m), nrow(m) == 200, ncol(m) == 200)
#' stopifnot(all(m >= 0 & m <= 1))
nlm_midpoint_displacement <- function(rows, cols, h = 1.0, seed = NULL) {
  r_midpoint_displacement(as.integer(rows), as.integer(cols),
                          as.double(h),
                          if (is.null(seed)) NULL else as.double(seed))
}

# ── Hill grow ─────────────────────────────────────────────────────────────────

#' Hill-grow NLM
#'
#' Iteratively applies a convolution kernel at randomly selected cells,
#' building up a surface of overlapping hills.
#'
#' @param rows      Number of rows.
#' @param cols      Number of columns.
#' @param n         Number of iterations (default 10 000).
#' @param runaway   If \code{TRUE}, selection probability is proportional to
#'   current cell height, causing hills to cluster (default \code{TRUE}).
#' @param kernel    A list of equal-length numeric vectors forming a square,
#'   odd-sized convolution kernel. \code{NULL} uses the default 3×3 diamond
#'   kernel.
#' @param only_grow If \code{TRUE}, the surface only accumulates; cells never
#'   shrink (default \code{FALSE}).
#' @param seed      Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_hill_grow(50, 50, n = 5000L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_hill_grow <- function(rows, cols, n = 10000L, runaway = TRUE,
                          kernel = NULL, only_grow = FALSE, seed = NULL) {
  r_hill_grow(as.integer(rows), as.integer(cols),
              as.integer(n),
              as.logical(runaway),
              kernel,
              as.logical(only_grow),
              if (is.null(seed)) NULL else as.double(seed))
}

# ── Perlin noise ──────────────────────────────────────────────────────────────

#' Perlin noise NLM
#'
#' Single-layer Perlin noise producing smooth, natural-looking gradients.
#'
#' @param rows         Number of rows.
#' @param cols         Number of columns.
#' @param scale_factor Noise frequency — higher values produce more features
#'   per unit area (default 4.0).
#' @param seed         Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_perlin_noise(50, 50, scale_factor = 4, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_perlin_noise <- function(rows, cols, scale_factor = 4.0, seed = NULL) {
  r_perlin_noise(as.integer(rows), as.integer(cols),
                 as.double(scale_factor),
                 if (is.null(seed)) NULL else as.double(seed))
}

# ── fBm noise ─────────────────────────────────────────────────────────────────

#' Fractal Brownian motion NLM
#'
#' Layers multiple octaves of Perlin noise for a more detailed, fractal result.
#'
#' @param rows         Number of rows.
#' @param cols         Number of columns.
#' @param scale_factor Base noise frequency (default 4.0).
#' @param octaves      Number of noise layers to combine (default 6).
#' @param persistence  Amplitude scaling per octave; \code{0.5} halves each
#'   successive layer (default 0.5).
#' @param lacunarity   Frequency scaling per octave; \code{2.0} doubles each
#'   successive layer (default 2.0).
#' @param seed         Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_fbm_noise(50, 50, octaves = 6L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_fbm_noise <- function(rows, cols, scale_factor = 4.0, octaves = 6L,
                          persistence = 0.5, lacunarity = 2.0, seed = NULL) {
  r_fbm_noise(as.integer(rows), as.integer(cols),
              as.double(scale_factor),
              as.integer(octaves),
              as.double(persistence),
              as.double(lacunarity),
              if (is.null(seed)) NULL else as.double(seed))
}

# ── Ridged noise ──────────────────────────────────────────────────────────────

#' Ridged multifractal NLM
#'
#' Produces sharp ridges and peak-like terrain. Similar to fBm but each
#' octave's noise value is folded, accumulating into pronounced ridges.
#'
#' @param rows        Number of rows.
#' @param cols        Number of columns.
#' @param scale_factor Base noise frequency (default 4.0).
#' @param octaves     Number of noise layers (default 6).
#' @param persistence Amplitude scaling per octave (default 0.5).
#' @param lacunarity  Frequency scaling per octave (default 2.0).
#' @param seed        Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_ridged_noise(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_ridged_noise <- function(rows, cols, scale_factor = 4.0, octaves = 6L,
                             persistence = 0.5, lacunarity = 2.0, seed = NULL) {
  r_ridged_noise(as.integer(rows), as.integer(cols),
                 as.double(scale_factor), as.integer(octaves),
                 as.double(persistence), as.double(lacunarity),
                 if (is.null(seed)) NULL else as.double(seed))
}

# ── Billow noise ───────────────────────────────────────────────────────────────

#' Billow NLM
#'
#' Billow noise applies an absolute-value fold to each octave of Perlin noise,
#' producing rounded, cloud- and hill-like patterns.
#'
#' @param rows        Number of rows.
#' @param cols        Number of columns.
#' @param scale_factor Base noise frequency (default 4.0).
#' @param octaves     Number of noise layers (default 6).
#' @param persistence Amplitude scaling per octave (default 0.5).
#' @param lacunarity  Frequency scaling per octave (default 2.0).
#' @param seed        Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_billow_noise(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_billow_noise <- function(rows, cols, scale_factor = 4.0, octaves = 6L,
                             persistence = 0.5, lacunarity = 2.0, seed = NULL) {
  r_billow_noise(as.integer(rows), as.integer(cols),
                 as.double(scale_factor), as.integer(octaves),
                 as.double(persistence), as.double(lacunarity),
                 if (is.null(seed)) NULL else as.double(seed))
}

# ── Worley noise ───────────────────────────────────────────────────────────────

#' Worley (cellular) noise NLM
#'
#' Each cell value is proportional to its distance to the nearest randomly
#' scattered Voronoi seed point, producing cellular / territory-like patches.
#'
#' @param rows        Number of rows.
#' @param cols        Number of columns.
#' @param scale_factor Seed-point frequency; higher values produce smaller
#'   cells (default 4.0).
#' @param seed        Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_worley_noise(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_worley_noise <- function(rows, cols, scale_factor = 4.0, seed = NULL) {
  r_worley_noise(as.integer(rows), as.integer(cols),
                 as.double(scale_factor),
                 if (is.null(seed)) NULL else as.double(seed))
}

# ── Gaussian field ─────────────────────────────────────────────────────────────

#' Gaussian random field NLM
#'
#' Generates white noise then applies a separable Gaussian blur, producing
#' spatially correlated surfaces where \code{sigma} directly controls the
#' ecological correlation length in cells.
#'
#' @param rows  Number of rows.
#' @param cols  Number of columns.
#' @param sigma Gaussian kernel standard deviation in cells (default 10.0).
#'   Higher values produce larger, smoother patches.
#' @param seed  Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_gaussian_field(50, 50, sigma = 5, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_gaussian_field <- function(rows, cols, sigma = 10.0, seed = NULL) {
  r_gaussian_field(as.integer(rows), as.integer(cols),
                   as.double(sigma),
                   if (is.null(seed)) NULL else as.double(seed))
}

# ── Random cluster ─────────────────────────────────────────────────────────────

#' Random cluster NLM
#'
#' Applies \code{n} random fault-line cuts; each cut adds +1 to cells on one
#' side and \eqn{-1} on the other. The accumulated field is scaled to
#' \eqn{[0, 1]}, producing clustered landscapes with linear structural elements.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of fault-line cuts (default 200). Higher values produce
#'   finer-grained clustering.
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_random_cluster(50, 50, n = 100L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_random_cluster <- function(rows, cols, n = 200L, seed = NULL) {
  r_random_cluster(as.integer(rows), as.integer(cols),
                   as.integer(n),
                   if (is.null(seed)) NULL else as.double(seed))
}

# ── Hybrid noise ──────────────────────────────────────────────────────────────

#' Hybrid multifractal NLM
#'
#' Blends smooth and ridged noise characteristics using the HybridMulti
#' combiner, producing terrain with varied spectral qualities.
#'
#' @param rows         Number of rows.
#' @param cols         Number of columns.
#' @param scale_factor Base noise frequency (default 4.0).
#' @param octaves      Number of noise layers (default 6).
#' @param persistence  Amplitude scaling per octave (default 0.5).
#' @param lacunarity   Frequency scaling per octave (default 2.0).
#' @param seed         Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_hybrid_noise(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_hybrid_noise <- function(rows, cols, scale_factor = 4.0, octaves = 6L,
                              persistence = 0.5, lacunarity = 2.0, seed = NULL) {
  r_hybrid_noise(as.integer(rows), as.integer(cols),
                 as.double(scale_factor), as.integer(octaves),
                 as.double(persistence), as.double(lacunarity),
                 if (is.null(seed)) NULL else as.double(seed))
}

# ── Value noise ────────────────────────────────────────────────────────────────

#' Value noise NLM
#'
#' Interpolated lattice noise producing blocky, low-frequency patterns.
#'
#' @param rows         Number of rows.
#' @param cols         Number of columns.
#' @param scale_factor Noise frequency — higher values produce more features
#'   per unit area (default 4.0).
#' @param seed         Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_value_noise(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_value_noise <- function(rows, cols, scale_factor = 4.0, seed = NULL) {
  r_value_noise(as.integer(rows), as.integer(cols),
                as.double(scale_factor),
                if (is.null(seed)) NULL else as.double(seed))
}

# ── Turbulence ─────────────────────────────────────────────────────────────────

#' Turbulence NLM
#'
#' Like fBm but accumulates the absolute value of each octave's contribution,
#' producing sharper, more chaotic textures.
#'
#' @param rows         Number of rows.
#' @param cols         Number of columns.
#' @param scale_factor Base noise frequency (default 4.0).
#' @param octaves      Number of noise layers (default 6).
#' @param persistence  Amplitude scaling per octave (default 0.5).
#' @param lacunarity   Frequency scaling per octave (default 2.0).
#' @param seed         Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_turbulence(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_turbulence <- function(rows, cols, scale_factor = 4.0, octaves = 6L,
                            persistence = 0.5, lacunarity = 2.0, seed = NULL) {
  r_turbulence(as.integer(rows), as.integer(cols),
               as.double(scale_factor), as.integer(octaves),
               as.double(persistence), as.double(lacunarity),
               if (is.null(seed)) NULL else as.double(seed))
}

# ── Domain warp ────────────────────────────────────────────────────────────────

#' Domain-warped Perlin noise NLM
#'
#' Displaces the sample coordinates of a base Perlin generator using a
#' secondary warp generator, producing highly organic, swirling patterns.
#'
#' @param rows          Number of rows.
#' @param cols          Number of columns.
#' @param scale_factor  Coordinate frequency — higher values produce more
#'   features per unit area (default 4.0).
#' @param warp_strength Displacement magnitude applied to sample coordinates
#'   (default 1.0).
#' @param seed          Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_domain_warp(50, 50, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_domain_warp <- function(rows, cols, scale_factor = 4.0, warp_strength = 1.0, seed = NULL) {
  r_domain_warp(as.integer(rows), as.integer(cols),
                as.double(scale_factor), as.double(warp_strength),
                if (is.null(seed)) NULL else as.double(seed))
}

# ── Mosaic ─────────────────────────────────────────────────────────────────────

#' Mosaic NLM
#'
#' Discrete Voronoi patch map. Places \code{n} seed cells each with a unique
#' random float value, then fills every remaining cell with the value of its
#' nearest seed. All cells within a patch share the same value, producing
#' flat-coloured regions.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of Voronoi seed points to place (default 200).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_mosaic(50, 50, n = 100L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_mosaic <- function(rows, cols, n = 200L, seed = NULL) {
  r_mosaic(as.integer(rows), as.integer(cols),
           as.integer(n),
           if (is.null(seed)) NULL else as.double(seed))
}

# ── Rectangular cluster ────────────────────────────────────────────────────────

#' Rectangular cluster NLM
#'
#' Places \code{n} random axis-aligned rectangles and accumulates +1 per cell
#' for each overlapping rectangle. The accumulated field is scaled to
#' \eqn{[0, 1]}, producing patch-like landscapes with rectilinear boundaries.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of rectangles to place (default 200).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_rectangular_cluster(50, 50, n = 100L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_rectangular_cluster <- function(rows, cols, n = 200L, seed = NULL) {
  r_rectangular_cluster(as.integer(rows), as.integer(cols),
                        as.integer(n),
                        if (is.null(seed)) NULL else as.double(seed))
}

# ── Percolation ───────────────────────────────────────────────────────────────

#' Percolation NLM
#'
#' Each cell is independently set to habitat (\code{1}) with probability
#' \code{p} and matrix (\code{0}) with probability \code{1 - p}.  As \code{p}
#' approaches the critical percolation threshold (~0.593 for 4-connectivity)
#' habitat clusters coalesce and span the landscape.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param p    Habitat probability in \eqn{[0, 1]} (default 0.5).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A binary numeric matrix with values in \eqn{\{0, 1\}}.
#' @export
#' @examples
#' m <- nlm_percolation(50, 50, p = 0.6, seed = 1L)
#' stopifnot(all(m %in% c(0, 1)))
nlm_percolation <- function(rows, cols, p = 0.5, seed = NULL) {
  r_percolation(as.integer(rows), as.integer(cols),
                as.double(p),
                if (is.null(seed)) NULL else as.double(seed))
}

# ── Binary space partitioning ─────────────────────────────────────────────────

#' Binary space partitioning NLM
#'
#' Recursively splits the grid into non-overlapping axis-aligned rectangles.
#' At each step the largest remaining rectangle is split along its longest
#' dimension at a random position. Each leaf rectangle is assigned a unique
#' random float, producing a hierarchically-nested rectilinear partition ideal
#' for modelling human-dominated agricultural or urban landscapes.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of rectangles in the final partition (default 100).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1)}.
#' @export
#' @examples
#' m <- nlm_binary_space_partitioning(50, 50, n = 20L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_binary_space_partitioning <- function(rows, cols, n = 100L, seed = NULL) {
  r_binary_space_partitioning(as.integer(rows), as.integer(cols),
                               as.integer(n),
                               if (is.null(seed)) NULL else as.double(seed))
}

#' Cellular automaton NLM
#'
#' Initialises a random binary grid and applies Conway-style birth/survival rules
#' for \code{iterations} steps. Out-of-bounds neighbours count as dead, producing
#' natural cave walls at edges.
#'
#' @param rows               Integer. Number of rows.
#' @param cols               Integer. Number of columns.
#' @param p                  Numeric. Initial alive probability (default 0.45).
#' @param iterations         Integer. Number of rule applications (default 5).
#' @param birth_threshold    Integer. Min live neighbours to birth a dead cell (default 5).
#' @param survival_threshold Integer. Min live neighbours for a live cell to survive (default 4).
#' @param seed               Optional integer seed for reproducible output.
#' @return A binary numeric matrix with values \code{0} or \code{1}.
#' @export
#' @examples
#' m <- nlm_cellular_automaton(50, 50, p = 0.45, seed = 1L)
#' stopifnot(all(m \%in\% c(0, 1)))
nlm_cellular_automaton <- function(rows, cols, p = 0.45, iterations = 5L,
                                    birth_threshold = 5L, survival_threshold = 4L,
                                    seed = NULL) {
  r_cellular_automaton(as.integer(rows), as.integer(cols),
                        as.double(p), as.integer(iterations),
                        as.integer(birth_threshold), as.integer(survival_threshold),
                        if (is.null(seed)) NULL else as.double(seed))
}

#' Neighbourhood clustering NLM
#'
#' Initialises a grid with \code{k} randomly assigned classes then repeatedly
#' applies a majority-vote rule: each cell adopts the most common class among
#' its 3x3 Moore neighbourhood. Produces smooth organic patch regions.
#'
#' @param rows       Integer. Number of rows.
#' @param cols       Integer. Number of columns.
#' @param k          Integer. Number of distinct patch classes (>= 2). Default 5.
#' @param iterations Integer. Number of majority-vote passes. Default 10.
#' @param seed       Optional integer seed for reproducible output.
#' @return A numeric matrix with values in \[0, 1).
#' @export
#' @examples
#' m <- nlm_neighbourhood_clustering(50, 50, k = 5L, iterations = 10L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_neighbourhood_clustering <- function(rows, cols, k = 5L, iterations = 10L, seed = NULL) {
  r_neighbourhood_clustering(as.integer(rows), as.integer(cols),
                              as.integer(k), as.integer(iterations),
                              if (is.null(seed)) NULL else as.double(seed))
}

#' Spectral synthesis NLM
#'
#' Generates correlated noise in the frequency domain. Each frequency component
#' is assigned a random phase and an amplitude proportional to
#' \eqn{f^{-\beta/2}}, giving a power spectrum \eqn{\propto 1/f^\beta}.
#'
#' @param rows Integer. Number of rows.
#' @param cols Integer. Number of columns.
#' @param beta Numeric. Spectral exponent. 0 = white noise, 1 = pink noise,
#'   2 = red/brown noise (natural terrain), higher = smoother. Default 2.0.
#' @param seed Optional integer seed for reproducible output.
#' @return A numeric matrix with values in \[0, 1).
#' @export
#' @examples
#' m <- nlm_spectral_synthesis(50, 50, beta = 2.0, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_spectral_synthesis <- function(rows, cols, beta = 2.0, seed = NULL) {
  r_spectral_synthesis(as.integer(rows), as.integer(cols),
                       as.double(beta),
                       if (is.null(seed)) NULL else as.double(seed))
}

# ── Reaction-diffusion ────────────────────────────────────────────────────────

#' Reaction-diffusion NLM (Gray-Scott model)
#'
#' Simulates two interacting chemicals (A and B) diffusing across the grid.
#' Different \code{feed}/\code{kill} combinations produce spots, stripes,
#' labyrinths, and other Turing-pattern morphologies.
#'
#' @param rows       Number of rows.
#' @param cols       Number of columns.
#' @param iterations Number of simulation steps (default 1000).
#' @param feed       Feed rate for chemical A (default 0.055).
#' @param kill       Kill rate for chemical B (default 0.062).
#' @param seed       Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_reaction_diffusion(30, 30, iterations = 100L, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_reaction_diffusion <- function(rows, cols, iterations = 1000L,
                                    feed = 0.055, kill = 0.062, seed = NULL) {
  r_reaction_diffusion(as.integer(rows), as.integer(cols),
                        as.integer(iterations),
                        as.double(feed), as.double(kill),
                        if (is.null(seed)) NULL else as.double(seed))
}

# ── Eden growth ────────────────────────────────────────────────────────────────

#' Eden growth model NLM
#'
#' Grows a compact cluster from the grid centre by repeatedly selecting a
#' random boundary cell and adding it. Produces irregular blob shapes with
#' fractal perimeters.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of cells to add to the initial centre seed (default 2000).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A binary numeric matrix with values in \eqn{\{0, 1\}}.
#' @export
#' @examples
#' m <- nlm_eden_growth(50, 50, n = 500L, seed = 1L)
#' stopifnot(all(m %in% c(0, 1)))
nlm_eden_growth <- function(rows, cols, n = 2000L, seed = NULL) {
  r_eden_growth(as.integer(rows), as.integer(cols),
                as.integer(n),
                if (is.null(seed)) NULL else as.double(seed))
}

# ── Fractal Brownian surface ───────────────────────────────────────────────────

#' Fractal Brownian surface NLM
#'
#' Generates a surface using the spectral synthesis method, parameterised
#' by the Hurst exponent \code{h}. Unlike \code{nlm_spectral_synthesis},
#' this function uses the ecologically meaningful Hurst exponent directly
#' (β = 2h + 2 internally).
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param h    Hurst exponent in (0, 1). Values near 0 are rough; values
#'   near 1 are smooth (default 0.5).
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_fractal_brownian_surface(50, 50, h = 0.7, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_fractal_brownian_surface <- function(rows, cols, h = 0.5, seed = NULL) {
  r_fractal_brownian_surface(as.integer(rows), as.integer(cols),
                              as.double(h),
                              if (is.null(seed)) NULL else as.double(seed))
}

# ── Landscape gradient ────────────────────────────────────────────────────────

#' Landscape gradient NLM
#'
#' Generates an elliptical gradient centred at the grid midpoint, decreasing
#' radially outward. The \code{direction} angle orients the major axis of the
#' ellipse, and \code{aspect} controls its elongation.
#'
#' @param rows      Number of rows.
#' @param cols      Number of columns.
#' @param direction Major-axis orientation in degrees \eqn{[0, 360)}.
#'   \code{NULL} picks a random direction.
#' @param aspect    Major-to-minor axis ratio (≥ 1.0). 1.0 = circular
#'   (default 1.0).
#' @param seed      Integer seed. \code{NULL} for random output.
#'
#' @return A numeric matrix with values in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_landscape_gradient(50, 50, direction = 45, aspect = 2.0, seed = 1L)
#' stopifnot(is.matrix(m))
nlm_landscape_gradient <- function(rows, cols, direction = NULL, aspect = 1.0, seed = NULL) {
  r_landscape_gradient(as.integer(rows), as.integer(cols),
                        if (is.null(direction)) NULL else as.double(direction),
                        as.double(aspect),
                        if (is.null(seed)) NULL else as.double(seed))
}

# ── Diffusion-limited aggregation ─────────────────────────────────────────────

#' Diffusion-limited aggregation NLM
#'
#' Grows a fractal cluster from the grid centre by releasing random-walking
#' particles that stick when they touch the existing cluster. Produces
#' intricate branching tree-like structures.
#'
#' @param rows Number of rows.
#' @param cols Number of columns.
#' @param n    Number of particles to release (default 2000). More particles
#'   produce a denser, more branching cluster.
#' @param seed Integer seed. \code{NULL} for random output.
#'
#' @return A binary numeric matrix with values in \eqn{\{0, 1\}}.
#' @export
#' @examples
#' m <- nlm_diffusion_limited_aggregation(50, 50, n = 500L, seed = 1L)
#' stopifnot(all(m %in% c(0, 1)))
nlm_diffusion_limited_aggregation <- function(rows, cols, n = 2000L, seed = NULL) {
  r_diffusion_limited_aggregation(as.integer(rows), as.integer(cols),
                                   as.integer(n),
                                   if (is.null(seed)) NULL else as.double(seed))
}

# ── Post-processing ───────────────────────────────────────────────────────────

#' Classify a landscape matrix into discrete classes
#'
#' Quantises each cell into one of \code{n} equal-width classes.
#' Class \eqn{k} (0-indexed) is assigned output value \eqn{k / (n - 1)},
#' evenly spacing the classes across \eqn{[0, 1]}.
#'
#' @param m A numeric matrix with values in \eqn{[0, 1]}, as returned by any
#'   \code{nlm_*} function.
#' @param n Number of equal-width classes (positive integer).
#'
#' @return A numeric matrix of the same dimensions with values at
#'   \code{n} discrete levels evenly spaced in \eqn{[0, 1]}.
#' @export
#' @examples
#' m <- nlm_midpoint_displacement(50, 50, seed = 1L)
#' cl <- nlm_classify(m, 4L)
#' stopifnot(length(unique(as.vector(cl))) <= 4L)
nlm_classify <- function(m, n) {
  r_classify(m, as.integer(n))
}

#' Threshold a landscape matrix to a binary habitat map
#'
#' Maps every cell to \code{0.0} if its value is strictly below \code{t},
#' or \code{1.0} otherwise. Useful for producing binary habitat/non-habitat maps.
#'
#' @param m A numeric matrix with values in \eqn{[0, 1]}.
#' @param t Threshold value. Cells below \code{t} become \code{0}; cells at or
#'   above become \code{1}.
#'
#' @return A binary numeric matrix with values \code{0} or \code{1}.
#' @export
#' @examples
#' m <- nlm_midpoint_displacement(50, 50, seed = 1L)
#' binary <- nlm_threshold(m, 0.5)
#' stopifnot(all(binary \%in\% c(0, 1)))
nlm_threshold <- function(m, t) {
  r_threshold(m, as.double(t))
}
