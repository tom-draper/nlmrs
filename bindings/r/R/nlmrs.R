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
