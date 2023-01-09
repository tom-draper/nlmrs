import csv

import matplotlib.pyplot as plt
import numpy as np


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
    
    plt.bar(range(0, len(data)), data, width=1.0, facecolor='black', edgecolor='black')
    plt.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    with open('./data/data.csv', 'r') as f:
        csvf = csv.reader(f)
    
        arr = []
        for lines in csvf:
            lines = list(map(float, lines))
            arr.append(lines)
        
        arr = np.array(arr)
        print(arr.shape)
        
        heatmap(arr)