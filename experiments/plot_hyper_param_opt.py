# %%

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable

if __name__ == "__main__":
    fig = plt.figure()
    data = pd.read_csv('stats/hyper_params.csv', index_col=False)
    grouped = data.groupby("sr hr skr".split()).mean()
    points = grouped.reset_index().values
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:, 3])
    cmap = plt.get_cmap("Spectral")
    norm = plt.Normalize(points[:, 3].min(), points[:, 3].max())
    sm = ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm)
    cbar.ax.set_title("Evaluation Metric")
    ax.set_xlabel("Scalar Vector weights")
    ax.set_ylabel("Distributional weights")
    ax.set_zlabel("Skeleton Feature weights")
    fig.tight_layout()
    plt.show()
