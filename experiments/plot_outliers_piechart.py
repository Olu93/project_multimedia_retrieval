import pandas as pd
from helper.config import STAT_PATH
import pandas as pd
import matplotlib.pyplot as plt
from helper.config import STAT_PATH
from collections import Counter


def mask_outlier(distribution):
    return [(distribution.mean() - 2 * distribution.std() < p < distribution.mean() + 2 * distribution.std()) for p in distribution]


def plot_outliers_piechart():
    df = pd.read_csv(STAT_PATH + "\\orig_stats.csv")
    vertices = df["vertices"]
    faces = df["faces"]
    cell_area_mean = df["cell_area_mean"]
    n_out_vertices = Counter(mask_outlier(vertices))
    n_out_faces = Counter(mask_outlier(faces))
    n_out_cell_area_mean = Counter(mask_outlier(cell_area_mean))

    fig, axes = plt.subplots(1, 3)

    axes[0].set_title("Faces")
    axes[0].pie(list(n_out_faces.values()), labels=["Inliers", "Outliers"], labeldistance=1.2)
    axes[1].set_title("Face Area")
    axes[1].pie(list(n_out_cell_area_mean.values()), labels=["Inliers", "Outliers"], labeldistance=1.2)
    axes[2].set_title("Vertices")
    axes[2].pie(list(n_out_vertices.values()), labels=["Inliers", "Outliers"], labeldistance=1.2)
    plt.show()


if __name__ == '__main__':
    plot_outliers_piechart()
