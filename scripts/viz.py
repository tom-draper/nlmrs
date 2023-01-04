import csv
import matplotlib.pyplot as plt

def heatmap(data: list[list[float]]):
    plt.figure(figsize=(8, 8))
    man = plt.get_current_fig_manager()
    man.canvas.set_window_title("Neutral Landscape Model")
    plt.pcolormesh(data, cmap='summer')
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
        
        heatmap(arr)