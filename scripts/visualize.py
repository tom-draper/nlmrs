import csv

import matplotlib.pyplot as plt
import numpy as np
import sys
import json


def heatmap(data: list[list[float]]):
    rows, cols = len(data), len(data[0])
    max_dim = max(rows, cols)
    scale = 10
    plt.figure(figsize=(scale * (cols / max_dim), scale * (rows / max_dim)))

    man = plt.get_current_fig_manager()
    man.canvas.manager.set_window_title("Neutral Landscape Model")

    plt.pcolormesh(data, cmap='terrain')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def sideview(data: list[float]):
    plt.figure(figsize=(10, 4))

    man = plt.get_current_fig_manager()
    man.canvas.manager.set_window_title("Neutral Landscape Model")

    plt.bar(range(0, len(data)), data, width=1.0,
            facecolor='black', edgecolor='black')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


def get_path() -> str:
    path = None
    if len(sys.argv) > 1:
        path = sys.argv[1]  # assume first command-line arg is file path
    return path


def get_file_extension(path: str) -> str:
    return path.split('.')[-1]


def parse_csv(path: str) -> list[list[float]]:
    arr = []
    with open(path, 'r') as f:
        csvf = csv.reader(f)

        for lines in csvf:
            lines = list(map(float, lines))
            arr.append(lines)

    return arr


def parse_json(path: str) -> list[list[float]]:
    arr = []
    with open(path) as f:
        arr = json.load(f)

    return arr


if __name__ == '__main__':
    path = get_path()
    if path is None:
        path = "./data/data.csv"  # default file path

    extension = get_file_extension(path)

    if extension == "csv":
        arr = parse_csv(path)
    elif extension == "json":
        arr = parse_json(path)
    else:
        exit()

    arr = np.array(arr)
    print(arr.shape)

    heatmap(arr)
