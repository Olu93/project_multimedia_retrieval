# %%
from collections import Counter
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean, sqeuclidean
from scipy.stats.stats import wasserstein_distance
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from matplotlib.cm import ScalarMappable

fig = plt.figure()
data = pd.read_csv('hyper_params.csv', index_col=False)
grouped = data.groupby("sr hr skr".split()).mean()
points = grouped.reset_index().values
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=points[:,3])
cmap = plt.get_cmap("Spectral")
norm = plt.Normalize(points[:,3].min(), points[:,3].max())
sm = ScalarMappable(norm=norm, cmap=cmap)
cbar = fig.colorbar(sm)
cbar.ax.set_title("Evaluation Metric")
ax.set_xlabel("Scalar Vector weights")
ax.set_ylabel("Distributional weights")
ax.set_zlabel("Skeleton Feature weights")
fig.tight_layout()
fig.show()
