# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "matplotlib",
#   "nlmrs",
#   "numpy",
# ]
#
# [tool.uv.sources]
# nlmrs = { path = ".." }
# ///
"""
Generate a PNG for every NLM algorithm using the nlmrs Python bindings.

Usage:
    uv run scripts/visualize.py [output_dir]

Output PNGs are written to `output_dir` (default: examples/).
uv builds the nlmrs extension automatically from the local source.
"""

import os
import sys

import matplotlib.pyplot as plt
import nlmrs
import numpy as np


SIZE = 100   # rows × cols for all grids
SEED = 42

# (filename_stem, fn_name, kwargs)
ALGORITHMS = [
    ("random",                "random",               {}),
    ("random_element",        "random_element",       {"n": 5000}),
    ("planar_gradient",       "planar_gradient",      {"direction": 45.0}),
    ("edge_gradient",         "edge_gradient",        {"direction": 45.0}),
    ("distance_gradient",     "distance_gradient",    {}),
    ("wave_gradient",         "wave_gradient",        {"period": 3.0}),
    ("midpoint_displacement", "midpoint_displacement",{"h": 0.8}),
    ("hill_grow",             "hill_grow",            {"n": 20000}),
    ("perlin",                "perlin_noise",         {"scale": 4.0}),
    ("fbm",                   "fbm_noise",            {"scale": 4.0, "octaves": 6}),
    ("ridged",                "ridged_noise",         {"scale": 4.0, "octaves": 6}),
    ("billow",                "billow_noise",         {"scale": 4.0, "octaves": 6}),
    ("worley",                "worley_noise",         {"scale": 4.0}),
    ("gaussian_field",        "gaussian_field",       {"sigma": 10.0}),
    ("random_cluster",        "random_cluster",       {"n": 200}),
    ("hybrid_noise",          "hybrid_noise",         {"scale": 4.0, "octaves": 6}),
    ("value_noise",           "value_noise",          {"scale": 4.0}),
    ("turbulence",            "turbulence",           {"scale": 4.0, "octaves": 6}),
    ("domain_warp",           "domain_warp",          {"scale": 4.0, "warp_strength": 1.0}),
    ("mosaic",                "mosaic",               {"n": 300}),
    ("rectangular_cluster",   "rectangular_cluster",  {"n": 300}),
    ("percolation",           "percolation",          {"p": 0.55}),
    ("binary_space_partitioning", "binary_space_partitioning", {"n": 200}),
    ("cellular_automaton",        "cellular_automaton",        {"p": 0.45, "iterations": 5}),
    ("neighbourhood_clustering",  "neighbourhood_clustering",  {"k": 5, "iterations": 10}),
    ("spectral_synthesis",        "spectral_synthesis",        {"beta": 2.0}),
    ("diffusion_limited_aggregation", "diffusion_limited_aggregation", {"n": 2000}),
    ("reaction_diffusion",       "reaction_diffusion",       {"iterations": 1000, "feed": 0.055, "kill": 0.062}),
    ("eden_growth",              "eden_growth",              {"n": 2000}),
    ("fractal_brownian_surface", "fractal_brownian_surface", {"h": 0.5}),
    ("landscape_gradient",       "landscape_gradient",       {"direction": 45.0, "aspect": 2.0}),
    ("simplex_noise",            "simplex_noise",            {"scale": 4.0}),
    ("invasion_percolation",     "invasion_percolation",     {"n": 2000}),
    ("gaussian_blobs",           "gaussian_blobs",           {"n": 50, "sigma": 5.0}),
    ("ising_model",              "ising_model",              {"beta": 0.4, "iterations": 1000}),
    ("gabor_noise",              "gabor_noise",              {"scale": 4.0, "n": 500}),
    ("spot_noise",               "spot_noise",               {"n": 200}),
    ("anisotropic_noise",        "anisotropic_noise",        {"scale": 4.0, "octaves": 6, "direction": 45.0, "stretch": 4.0}),
    ("tiled_noise",              "tiled_noise",              {"scale": 4.0}),
    ("brownian_motion",          "brownian_motion",          {"n": 5000}),
    ("forest_fire",              "forest_fire",              {"p_tree": 0.02, "p_lightning": 0.001, "iterations": 500}),
    ("river_network",            "river_network",            {}),
    ("hexagonal_voronoi",        "hexagonal_voronoi",        {"n": 50}),
    ("voronoi_distance",         "voronoi_distance",         {"n": 50}),
    ("sine_composite",           "sine_composite",           {"waves": 8}),
    ("curl_noise",               "curl_noise",               {"scale": 4.0}),
    ("hydraulic_erosion",        "hydraulic_erosion",        {"n": 500}),
    ("levy_flight",              "levy_flight",              {"n": 1000}),
    ("poisson_disk",             "poisson_disk",             {"min_dist": 5.0}),
]


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_png(grid: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(grid, cmap="terrain", vmin=0, vmax=1)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    fig.savefig(out_path, dpi=150, pad_inches=0)
    plt.close(fig)


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "examples")
    os.makedirs(out_dir, exist_ok=True)

    for stem, fn_name, kwargs in ALGORITHMS:
        png_path = os.path.join(out_dir, f"{stem}.png")
        title    = stem.replace("_", " ").title()

        print(f"  {title:<28}", end="", flush=True)
        fn = getattr(nlmrs, fn_name)
        grid = fn(SIZE, SIZE, **kwargs, seed=SEED)
        save_png(grid, png_path)
        print(f"→ {os.path.relpath(png_path)}")

    print(f"\nDone. {len(ALGORITHMS)} PNGs saved to {out_dir}/")


if __name__ == "__main__":
    main()
