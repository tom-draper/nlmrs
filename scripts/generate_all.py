# /// script
# requires-python = ">=3.8"
# dependencies = [
#   "matplotlib",
#   "numpy",
# ]
# ///
"""
Generate a PNG for every NLM algorithm.

Usage:
    python3 scripts/generate_all.py [output_dir]

Output PNGs are written to `output_dir` (default: output/algorithms/).
The release binary is built automatically if not present.
"""

import csv
import os
import subprocess
import sys
import tempfile

import matplotlib.pyplot as plt
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────────────

SIZE = 400   # rows × cols for all grids
SEED = 42

ALGORITHMS = [
    # (filename_stem, cli_subcommand, extra_args)
    ("random",                "random",               []),
    ("random_element",        "random-element",       ["--n", "5000"]),
    ("planar_gradient",       "planar-gradient",      ["--direction", "45"]),
    ("edge_gradient",         "edge-gradient",        ["--direction", "45"]),
    ("distance_gradient",     "distance-gradient",    []),
    ("wave_gradient",         "wave-gradient",        ["--period", "3"]),
    ("midpoint_displacement", "midpoint-displacement",["--h", "0.8"]),
    ("hill_grow",             "hill-grow",            ["--n", "20000"]),
    ("perlin",                "perlin",               ["--scale", "4"]),
    ("fbm",                   "fbm",                  ["--scale", "4", "--octaves", "6"]),
    ("ridged",                "ridged",               ["--scale", "4", "--octaves", "6"]),
    ("billow",                "billow",               ["--scale", "4", "--octaves", "6"]),
    ("worley",                "worley",               ["--scale", "4"]),
    ("gaussian_field",        "gaussian-field",       ["--sigma", "10"]),
    ("random_cluster",        "random-cluster",       ["--n", "200"]),
    ("hybrid_noise",          "hybrid-noise",         ["--scale", "4", "--octaves", "6"]),
    ("value_noise",           "value-noise",          ["--scale", "4"]),
    ("turbulence",            "turbulence",           ["--scale", "4", "--octaves", "6"]),
    ("domain_warp",           "domain-warp",          ["--scale", "4", "--warp-strength", "1.0"]),
    ("mosaic",                "mosaic",               ["--n", "300"]),
    ("rectangular_cluster",   "rectangular-cluster",  ["--n", "300"]),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BINARY    = os.path.join(REPO_ROOT, "target", "release", "nlmrs")


def ensure_binary():
    if not os.path.exists(BINARY):
        print("Release binary not found — building...")
        subprocess.run(
            ["cargo", "build", "--release"],
            cwd=REPO_ROOT,
            check=True,
        )


def run_algorithm(subcommand: str, extra_args: list, csv_path: str):
    cmd = [
        BINARY,
        subcommand,
        str(SIZE), str(SIZE),
        *extra_args,
        "--seed", str(SEED),
        "--output", csv_path,
    ]
    subprocess.run(cmd, check=True, capture_output=True)


def read_csv(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        rows = [[float(v) for v in row] for row in csv.reader(f)]
    return np.array(rows)


def save_png(grid: np.ndarray, out_path: str):
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pcolormesh(grid, cmap="terrain", vmin=0, vmax=1)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")
    fig.savefig(out_path, dpi=150, pad_inches=0)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO_ROOT, "output", "algorithms")
    os.makedirs(out_dir, exist_ok=True)

    ensure_binary()

    with tempfile.TemporaryDirectory() as tmp:
        for stem, subcommand, extra in ALGORITHMS:
            csv_path = os.path.join(tmp, f"{stem}.csv")
            png_path = os.path.join(out_dir, f"{stem}.png")
            title    = stem.replace("_", " ").title()

            print(f"  {title:<28}", end="", flush=True)
            run_algorithm(subcommand, extra, csv_path)
            grid = read_csv(csv_path)
            save_png(grid, png_path)
            print(f"→ {os.path.relpath(png_path)}")

    print(f"\nDone. {len(ALGORITHMS)} PNGs saved to {out_dir}/")


if __name__ == "__main__":
    main()
