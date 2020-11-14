# %%
from collections import Counter
from sys import path
import os.path
from matplotlib.pyplot import title
from matplotlib.lines import Line2D
import sys

from pandas.core.frame import DataFrame
import json
from pandas.io.pytables import Selection
from scipy.sparse.construct import random
from query_matcher import QueryMatcher
from helper.misc import rand_cmap
import io
import jsonlines
from scipy.spatial.distance import cityblock, cosine, euclidean
from scipy.stats.stats import wasserstein_distance
from tqdm.std import tqdm
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE
from evaluator import Evaluator
import pandas as pd
import itertools
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.colors as cl
import numpy as np
import multiprocessing as mp
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib as mpl
from sklearn.utils import resample
from random import sample
from pprint import pprint

metrics_options = [
    'cityblock', 'cosine', 'euclidean', 'haversine', 'l2', 'l1', 'manhattan', 'precomputed', 'nan_euclidean', 'braycurtis', 'canberra', 'chebyshev', 'correlation', 'cosine',
    'dice', 'hamming', 'jaccard', 'kulsinski', 'mahalanobis', 'matching', 'minkowski', 'rogerstanimoto', 'russellrao', 'seuclidean', 'sokalmichener', 'sokalsneath', 'sqeuclidean',
    'yule', 'wminkowski'
]
VERBOSE = True
NUM_JOBS = 6

def compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, is_coarse=False, d=2, num_jobs=None) -> pd.DataFrame:
    print(f" Computing... {constuct_identifier(perplexity= perplexity, n_iter = n_iter, lr= lr)}")
    if n_iter != -1:
        tsne_results = TSNE(d, perplexity=perplexity, early_exaggeration=1, n_iter=int(n_iter), learning_rate=lr, n_jobs=NUM_JOBS, verbose=VERBOSE).fit_transform(flat_data)
    else:
        tsne_results = TSNE(d, perplexity=5, early_exaggeration=1, n_iter=int(10e7), learning_rate=1, n_jobs=6, verbose=1).fit_transform(flat_data)

    all_together = np.array([x + list(y) for x, y in zip(names_n_labels, tsne_results)], dtype=object)
    cols = ["label", "x", "y"]
    if d == 3:
        cols = ["label", "x", "y", "z"]
    if d > 3:
        cols = ["label"] + [f"x{idx}" for idx in range(d)]
    df = pd.DataFrame(all_together[:, 1:], index=all_together[:, 0], columns=cols)
    return df


def run_experiment(query_matcher, params, iterations, lr_rates):
    num_experiments = len(params)
    flat_data = query_matcher.features_df
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig, axes = plt.subplots(num_experiments // 2, 2, figsize=figsize) if (num_experiments % 2) == 0 else plt.subplots((num_experiments // 3) + 1, 3, figsize=figsize)
    flattened_axes = axes.ravel()

    scatters = []
    for ax, param, num_iterations, lr in zip(flattened_axes, params, iterations, lr_rates):
        tsne_result = compute_tsne(flat_data, param, num_iterations, lr, names_n_labels, True)
        label_ids = [classification_indexes[label] for label in tsne_result.label]
        cmap = plt.cm.get_cmap("hsv", len(unique_label_ids))
        scatters.append(ax.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap))
        ax.set_title(f"Perplexity {param}, Iterations {num_iterations}")

    fig.tight_layout()
    cbar = fig.colorbar(scatters[-1], ticks=unique_label_ids, ax=flattened_axes.tolist())
    cbar.ax.set_yticklabels(unique_labels)
    return fig


def constuct_identifier(idx="X", perplexity="X", n_iter="X", lr="X"):
    return f"{idx}_{int(perplexity)}_{int(lr)}_{n_iter}"


def prepare_data(query_matcher, top_n=0, reverse=False, strange_scaling=False, is_coarse=True):
    flat_data = query_matcher.features_df_properly_scaled
    all_lbls = [query_matcher.map_to_label(name, is_coarse) for name in list(flat_data.index)]
    final_data = flat_data

    if strange_scaling:
        flat_data = query_matcher.features_df_all_scaled
    if top_n:
        flat_data["label"] = all_lbls
        all_lbls_cnt = Counter(all_lbls)
        # resversed_order_bit = -1 if reverse else 1
        lbl_count_df = pd.DataFrame(all_lbls_cnt.most_common(), columns=("label", "cnt"))
        sum_top_lbls = lbl_count_df[:top_n].cnt.sum()
        ratio = sum_top_lbls / flat_data.shape[0]
        cutoff_point = int(ratio * lbl_count_df.cnt.sum())

        top_lbls = {}
        selected_lbls = None
        for key, cnt in lbl_count_df.sort_values(by="cnt", ascending=reverse).values:

            top_lbls[key] = cnt
            if sum(top_lbls.values()) >= cutoff_point:
                selected_lbls = pd.DataFrame(top_lbls.items(), columns=("label", "cnt"))
                break
        # lbl_count_df.sort_values(by="cnt", ascending=reverse).iloc[:cutoff_point]
        min_lbl_cnt = selected_lbls.cnt.min()

        final_data = pd.concat([flat_data[flat_data.label == key].sample(min_lbl_cnt) for key, _ in selected_lbls.values]).drop("label", axis=1)

    return final_data


def show_all_in_one(query_matcher, idx, perplexity, n_iter, lr, show=False, top_n=False, reverse=False, strange_scaling=False, is_coarse=True, prod=False):

    # Data section
    data = prepare_data(query_matcher, top_n=top_n, reverse=reverse, strange_scaling=strange_scaling, is_coarse=is_coarse)
    print(data.shape)
    # balanced_flat_data =
    pca_data = pd.DataFrame(PCA(n_components=50).fit_transform(data), index=data.index)
    # pca_data["label"] = [query_matcher.map_to_label(name, True) for name in list(pca_data.index)]
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(pca_data.index)]
    id_to_class = dict(enumerate(sorted(set([label for _, label in names_n_labels]))))
    class_to_id = {label: idx for idx, label in id_to_class.items()}

    unique_labels, unique_label_ids = zip(*class_to_id.items())
    # tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, is_coarse=True, d=2, num_jobs=None)
    tsne_result_pca = compute_tsne(pca_data, perplexity, n_iter, lr, names_n_labels, is_coarse=True, d=2, num_jobs=None)
    tsne_result_3d = None
    if prod:
        tsne_result_3d = compute_tsne(pca_data, perplexity, n_iter, lr, names_n_labels, is_coarse=True, d=3, num_jobs=None)
    label_ids = np.array([class_to_id[label] for label in tsne_result_pca.label])
    tsne_result_pca["lbl_id"] = label_ids
    # tsne_result_3d["lbl_id"] = label_ids

    figsize = (16, 8)
    figdpi = 80
    msize = 10
    num_subset_views = 3
    fig, axes = plt.subplots(2, 3, num=idx, figsize=figsize, dpi=figdpi, sharex=True, sharey=True)
    ax = axes.ravel()
    cmap = plt.cm.get_cmap("nipy_spectral", len(unique_label_ids))
    norm = mpl.colors.Normalize(vmin=0, vmax=max(unique_label_ids))
    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)

    idx = 0
    # PCA
    ax[idx].set_title("t-SNE with PCA to 50")
    ax[idx].scatter("x", "y", c="lbl_id", s=msize, cmap=cmap, data=tsne_result_pca)

    # Full image
    idx = 1
    chosen_triplets = np.array([class_to_id.get(lbl) for lbl in ("plant", "handheld", "car") if lbl in class_to_id])
    tmp_ax = ax[idx]
    lbl_wi_color = np.isin(label_ids, chosen_triplets)
    tmp_ax.scatter("x", "y", s=1, c="w", cmap=cmap, data=tsne_result_pca[np.isin(label_ids, chosen_triplets) != 1])
    tmp_ax.scatter("x", "y", s=msize * 1, c="lbl_id", cmap=cmap, data=tsne_result_pca[lbl_wi_color == 1])
    legend_elements = [Line2D([0], [0], marker='o', color='w', label=id_to_class[lbl_id], markerfacecolor=cmap(norm(lbl_id)), markersize=5) for lbl_id in chosen_triplets]
    tmp_ax.set_title(f'fixed ({", ".join([id_to_class[lbl_id] for lbl_id in chosen_triplets])})', fontsize=12)
    tmp_ax.legend(handles=legend_elements, loc="upper right")
    # cmap = plt.cm.get_cmap("hsv", len(set(label_ids)) + 1)
    num_unique_lbls = len(unique_label_ids)
    num_pick = min(num_unique_lbls // num_subset_views, 3)
    skip = 3 if num_unique_lbls > 27 else 1
    rand_choice = np.random.choice(np.array(unique_label_ids)[::skip], (3, num_pick), replace=False)
    for idx, chosen_triplets in zip(range(3, len(ax)), rand_choice):
        tmp_ax = ax[idx]
        lbl_wi_color = np.isin(label_ids, chosen_triplets)
        tmp_ax.scatter("x", "y", s=1, c="w", cmap=cmap, data=tsne_result_pca[np.isin(label_ids, rand_choice) != 1])
        tmp_ax.scatter("x", "y", s=msize * 1, c="lbl_id", cmap=cmap, data=tsne_result_pca[lbl_wi_color == 1])
        legend_elements = [Line2D([0], [0], marker='o', color='w', label=id_to_class[lbl_id], markerfacecolor=cmap(norm(lbl_id)), markersize=5) for lbl_id in chosen_triplets]
        tmp_ax.set_title(f'{idx-2} ({", ".join([id_to_class[lbl_id] for lbl_id in chosen_triplets])})', fontsize=12)
        tmp_ax.legend(handles=legend_elements, loc="upper right")

    fig.suptitle(f"Perplexity ({perplexity:.2f}), Iterations ({n_iter:.1E}), LR ({lr})")

    idx = 2
    # 3D Image
    if prod:
        ax[idx].remove()
        ax[idx] = fig.add_subplot(2, 3, 3, projection="3d")
        ax[idx].set_title("t-SNE in 3D")
        ax[idx].scatter(xs=tsne_result_3d.x, ys=tsne_result_3d.y, zs=tsne_result_3d.z, s=msize, c=tsne_result_3d.lbl_id, cmap=cmap)

    # Align color map with a

    plt.tight_layout()

    for scat in itertools.chain(*[ax_i.collections for ax_i in ax]):
        scat.set_cmap(cmap)
        scat.set_norm(norm)
    cbar = fig.colorbar(sm, ticks=unique_label_ids, ax=ax.tolist())
    cbar.ax.set_yticklabels(unique_labels)
    # fig.canvas.draw()
    # fig.canvas.flush_events()
    if show:
        return plt.show()
    return fig


def save_all_in_one(params):
    _ = show_all_in_one(*params[:-1])
    plt.savefig(params[-1])
    plt.close()
    print("Done ", params[1])


def run_all_in_one_experiment(query_matcher, workers=1):
    perplexity_range = np.linspace(1, 101, 100)
    iterations_range = [int(1e6)]
    learning_rates_range = [1]

    combinations = []
    for idx in perplexity_range:
        pr = idx
        ir = np.random.choice(iterations_range)
        lrr = np.random.choice(learning_rates_range)
        identifier = constuct_identifier("tsne", pr, ir, lrr)
        file_name = f"trash/tsne_full_5_top10_coarse_grained/{identifier}.png"
        combinations.append((query_matcher, idx, pr, ir, lrr, False, 10, False, False, True, file_name))

    if workers == 1:
        for params in tqdm(set(combinations), total=len(combinations)):
            if os.path.exists(params[-1]):
                continue
            save_all_in_one(params)

    if workers > 1:
        pool = mp.Pool(workers)
        results = pool.imap(save_all_in_one, tqdm(set(combinations), total=len(combinations)))
        _ = list(results)


if __name__ == "__main__":
    query_matcher = QueryMatcher(FEATURE_DATA_FILE)
    arguments = sys.argv[1:]

    if "-p" in arguments:
        run_all_in_one_experiment(query_matcher, 6)

    if "-p" not in arguments:
        iters = int(20e6)
        show_all_in_one(query_matcher, 0, 5, -1, 1, show=True, top_n=15, reverse=True, strange_scaling=False, is_coarse=False)
    # lbl_cnts = [query_matcher.map_to_label(name, True) for name in list(query_matcher.features_df_all_scaled.index)]

    # run_parallel_exploration_3D(query_matcher)
    # run_parallel_exploration_scaled(query_matcher)
