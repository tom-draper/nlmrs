module nlmrs

const _deps = joinpath(@__DIR__, "..", "deps", "deps.jl")
isfile(_deps) ||
    error("nlmrs not built. Run `import Pkg; Pkg.build(\"nlmrs\")`.")
include(_deps)  # defines _libpath

# ── Internal C struct ─────────────────────────────────────────────────────────

struct _NlmGrid
    data::Ptr{Float64}
    rows::Csize_t
    cols::Csize_t
end

# ── Helpers ───────────────────────────────────────────────────────────────────

function _to_matrix(grid::_NlmGrid)::Matrix{Float64}
    rows = Int(grid.rows)
    cols = Int(grid.cols)
    # Rust data is row-major; Julia matrices are column-major.
    # Reshape as (cols, rows) then transpose to get a proper (rows, cols) Matrix.
    flat = unsafe_wrap(Array, grid.data, rows * cols)
    mat  = collect(reshape(flat, cols, rows)')
    ccall((:nlmrs_free, _libpath), Cvoid, (_NlmGrid,), grid)
    return mat
end

_seed(::Nothing)   = C_NULL
_seed(s::Integer)  = Ref(UInt64(s))
_f64(::Nothing)    = C_NULL
_f64(v::Real)      = Ref(Cdouble(v))

# ── Exports ───────────────────────────────────────────────────────────────────

export planar_gradient, edge_gradient, distance_gradient, wave_gradient,
       landscape_gradient,
       perlin_noise, value_noise, worley_noise, fbm_noise, ridged_noise,
       billow_noise, hybrid_noise, turbulence, domain_warp, spectral_synthesis,
       fractal_brownian_surface, simplex_noise, voronoi_distance, sine_composite,
       curl_noise, gabor_noise, spot_noise, anisotropic_noise, tiled_noise,
       random, random_element, hill_grow, midpoint_displacement, random_cluster,
       mosaic, rectangular_cluster, percolation, binary_space_partitioning,
       cellular_automaton, neighbourhood_clustering, reaction_diffusion,
       eden_growth, diffusion_limited_aggregation, invasion_percolation,
       gaussian_blobs, ising_model, hydraulic_erosion, levy_flight, poisson_disk,
       gaussian_field, brownian_motion, forest_fire, river_network,
       hexagonal_voronoi

# ── Gradient ──────────────────────────────────────────────────────────────────

function planar_gradient(rows::Integer, cols::Integer;
                         direction=nothing, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_planar_gradient, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{Cdouble}, Ptr{UInt64}),
        rows, cols, _f64(direction), _seed(seed)))
end

function edge_gradient(rows::Integer, cols::Integer;
                       direction=nothing, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_edge_gradient, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{Cdouble}, Ptr{UInt64}),
        rows, cols, _f64(direction), _seed(seed)))
end

function distance_gradient(rows::Integer, cols::Integer;
                            seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_distance_gradient, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, _seed(seed)))
end

function wave_gradient(rows::Integer, cols::Integer;
                       period::Real=2.5, direction=nothing,
                       seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_wave_gradient, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{Cdouble}, Ptr{UInt64}),
        rows, cols, period, _f64(direction), _seed(seed)))
end

function landscape_gradient(rows::Integer, cols::Integer;
                             direction=nothing, aspect::Real=1.0,
                             seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_landscape_gradient, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{Cdouble}, Cdouble, Ptr{UInt64}),
        rows, cols, _f64(direction), aspect, _seed(seed)))
end

# ── Noise ─────────────────────────────────────────────────────────────────────

function perlin_noise(rows::Integer, cols::Integer;
                      scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_perlin_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

function value_noise(rows::Integer, cols::Integer;
                     scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_value_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

function worley_noise(rows::Integer, cols::Integer;
                      scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_worley_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

function fbm_noise(rows::Integer, cols::Integer;
                   scale::Real=4.0, octaves::Integer=6,
                   persistence::Real=0.5, lacunarity::Real=2.0,
                   seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_fbm_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, persistence, lacunarity, _seed(seed)))
end

function ridged_noise(rows::Integer, cols::Integer;
                      scale::Real=4.0, octaves::Integer=6,
                      persistence::Real=0.5, lacunarity::Real=2.0,
                      seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_ridged_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, persistence, lacunarity, _seed(seed)))
end

function billow_noise(rows::Integer, cols::Integer;
                      scale::Real=4.0, octaves::Integer=6,
                      persistence::Real=0.5, lacunarity::Real=2.0,
                      seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_billow_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, persistence, lacunarity, _seed(seed)))
end

function hybrid_noise(rows::Integer, cols::Integer;
                      scale::Real=4.0, octaves::Integer=6,
                      persistence::Real=0.5, lacunarity::Real=2.0,
                      seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_hybrid_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, persistence, lacunarity, _seed(seed)))
end

function turbulence(rows::Integer, cols::Integer;
                    scale::Real=4.0, octaves::Integer=6,
                    persistence::Real=0.5, lacunarity::Real=2.0,
                    seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_turbulence, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, persistence, lacunarity, _seed(seed)))
end

function domain_warp(rows::Integer, cols::Integer;
                     scale::Real=4.0, warp_strength::Real=1.0,
                     seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_domain_warp, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, warp_strength, _seed(seed)))
end

function spectral_synthesis(rows::Integer, cols::Integer;
                             beta::Real=2.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_spectral_synthesis, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, beta, _seed(seed)))
end

function fractal_brownian_surface(rows::Integer, cols::Integer;
                                   h::Real=0.5, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_fractal_brownian_surface, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, h, _seed(seed)))
end

function simplex_noise(rows::Integer, cols::Integer;
                       scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_simplex_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

function voronoi_distance(rows::Integer, cols::Integer;
                           n::Integer=50, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_voronoi_distance, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function sine_composite(rows::Integer, cols::Integer;
                         waves::Integer=8, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_sine_composite, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, waves, _seed(seed)))
end

function curl_noise(rows::Integer, cols::Integer;
                    scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_curl_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

function gabor_noise(rows::Integer, cols::Integer;
                     scale::Real=4.0, n::Integer=500,
                     seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_gabor_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Ptr{UInt64}),
        rows, cols, scale, n, _seed(seed)))
end

function spot_noise(rows::Integer, cols::Integer;
                    n::Integer=200, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_spot_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function anisotropic_noise(rows::Integer, cols::Integer;
                            scale::Real=4.0, octaves::Integer=6,
                            direction::Real=45.0, stretch::Real=4.0,
                            seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_anisotropic_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, scale, octaves, direction, stretch, _seed(seed)))
end

function tiled_noise(rows::Integer, cols::Integer;
                     scale::Real=4.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_tiled_noise, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, scale, _seed(seed)))
end

# ── Patch ─────────────────────────────────────────────────────────────────────

function random(rows::Integer, cols::Integer;
                seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_random, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, _seed(seed)))
end

function random_element(rows::Integer, cols::Integer;
                         n::Real=50000.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_random_element, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, Float64(n), _seed(seed)))
end

function hill_grow(rows::Integer, cols::Integer;
                   n::Integer=10000, runaway::Bool=true,
                   kernel::Union{Matrix{Float64}, Nothing}=nothing,
                   only_grow::Bool=false, seed=nothing)::Matrix{Float64}
    if isnothing(kernel)
        kdata = C_NULL
        ksize = Csize_t(0)
    else
        size(kernel, 1) == size(kernel, 2) || error("kernel must be square")
        ksize = Csize_t(size(kernel, 1))
        kdata = vec(permutedims(kernel))  # Julia column-major → row-major for C
    end
    _to_matrix(ccall((:nlmrs_hill_grow, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Bool, Ptr{Cdouble}, Csize_t, Bool, Ptr{UInt64}),
        rows, cols, n, runaway, kdata, ksize, only_grow, _seed(seed)))
end

function midpoint_displacement(rows::Integer, cols::Integer;
                                h::Real=1.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_midpoint_displacement, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, h, _seed(seed)))
end

function random_cluster(rows::Integer, cols::Integer;
                         n::Integer=200, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_random_cluster, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function mosaic(rows::Integer, cols::Integer;
                n::Integer=200, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_mosaic, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function rectangular_cluster(rows::Integer, cols::Integer;
                               n::Integer=200, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_rectangular_cluster, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function percolation(rows::Integer, cols::Integer;
                     p::Real=0.5, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_percolation, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, p, _seed(seed)))
end

function binary_space_partitioning(rows::Integer, cols::Integer;
                                    n::Integer=100, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_binary_space_partitioning, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function cellular_automaton(rows::Integer, cols::Integer;
                             p::Real=0.45, iterations::Integer=5,
                             birth_threshold::Integer=5,
                             survival_threshold::Integer=4,
                             seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_cellular_automaton, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, p, iterations, birth_threshold, survival_threshold, _seed(seed)))
end

function neighbourhood_clustering(rows::Integer, cols::Integer;
                                   k::Integer=5, iterations::Integer=10,
                                   seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_neighbourhood_clustering, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, k, iterations, _seed(seed)))
end

function reaction_diffusion(rows::Integer, cols::Integer;
                             iterations::Integer=1000, feed::Real=0.055,
                             kill::Real=0.062, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_reaction_diffusion, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Cdouble, Cdouble, Ptr{UInt64}),
        rows, cols, iterations, feed, kill, _seed(seed)))
end

function eden_growth(rows::Integer, cols::Integer;
                     n::Integer=2000, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_eden_growth, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function diffusion_limited_aggregation(rows::Integer, cols::Integer;
                                        n::Integer=2000, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_diffusion_limited_aggregation, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function invasion_percolation(rows::Integer, cols::Integer;
                               n::Integer=2000, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_invasion_percolation, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function gaussian_blobs(rows::Integer, cols::Integer;
                         n::Integer=50, sigma::Real=5.0,
                         seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_gaussian_blobs, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, n, sigma, _seed(seed)))
end

function ising_model(rows::Integer, cols::Integer;
                     beta::Real=0.4, iterations::Integer=1000,
                     seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_ising_model, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Csize_t, Ptr{UInt64}),
        rows, cols, beta, iterations, _seed(seed)))
end

function hydraulic_erosion(rows::Integer, cols::Integer;
                            n::Integer=500, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_hydraulic_erosion, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function levy_flight(rows::Integer, cols::Integer;
                     n::Integer=1000, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_levy_flight, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function poisson_disk(rows::Integer, cols::Integer;
                      min_dist::Real=5.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_poisson_disk, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, min_dist, _seed(seed)))
end

function gaussian_field(rows::Integer, cols::Integer;
                         sigma::Real=10.0, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_gaussian_field, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Ptr{UInt64}),
        rows, cols, sigma, _seed(seed)))
end

function brownian_motion(rows::Integer, cols::Integer;
                          n::Integer=5000, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_brownian_motion, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

function forest_fire(rows::Integer, cols::Integer;
                     p_tree::Real=0.02, p_lightning::Real=0.001,
                     iterations::Integer=500, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_forest_fire, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Cdouble, Cdouble, Csize_t, Ptr{UInt64}),
        rows, cols, p_tree, p_lightning, iterations, _seed(seed)))
end

function river_network(rows::Integer, cols::Integer;
                        seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_river_network, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, _seed(seed)))
end

function hexagonal_voronoi(rows::Integer, cols::Integer;
                            n::Integer=50, seed=nothing)::Matrix{Float64}
    _to_matrix(ccall((:nlmrs_hexagonal_voronoi, _libpath), _NlmGrid,
        (Csize_t, Csize_t, Csize_t, Ptr{UInt64}),
        rows, cols, n, _seed(seed)))
end

end # module nlmrs
