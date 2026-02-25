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


SIZE = 400   # rows × cols for all grids
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
