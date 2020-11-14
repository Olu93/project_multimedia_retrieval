from collections import Counter
from sys import path
import os.path
from matplotlib.pyplot import title
from matplotlib.lines import Line2D

from pandas.core.frame import DataFrame
from pandas.io import json
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


def compute_tsne_3D(flat_data, perplexity, n_iter, lr, names_n_labels, is_coarse=False) -> pd.DataFrame:
    print(f"Computing... {constuct_identifier(perplexity= perplexity, n_iter = n_iter, lr= lr)}")
    tsne_results = TSNE(3, perplexity=perplexity, n_iter=n_iter, learning_rate=lr).fit_transform(flat_data)
    all_together = np.array([x + list(y) for x, y in zip(names_n_labels, tsne_results)], dtype=object)
    df = pd.DataFrame(all_together[:, 1:], index=all_together[:, 0], columns=["label", "x", "y", "z"])
    return df


def show_tsne(query_matcher, idx, perplexity, n_iter, lr):
    flat_data = query_matcher.features_df_properly_scaled
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    # ax = fig.subplot()

    tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("nipy_spectral", len(unique_label_ids))
    scatter = plt.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)
    plt.tight_layout()

    plt.show()


def show_tsne_3D(query_matcher, idx, perplexity, n_iter, lr):
    flat_data = query_matcher.features_df_properly_scaled
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    tsne_result = compute_tsne_3D(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("hsv", len(unique_label_ids))
    scatter = ax.scatter(tsne_result.x.values, tsne_result.y.values, tsne_result.z.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)
    plt.tight_layout()

    plt.show()


def show_tsne_label_subset(query_matcher, selected_labels, idx, perplexity, n_iter, lr):
    flat_data = query_matcher.features_df_properly_scaled
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    # ax = fig.subplot()

    tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    label_ids = np.array([selected_labels.index(lbl_id) if lbl_id in selected_labels else -1 for lbl_id in label_ids])
    lbl_nocl = label_ids == -1
    lbl_wicl = label_ids != -1
    cmap = plt.cm.get_cmap("hsv", len(set(label_ids)) + 1)
    scatter = plt.scatter(tsne_result[lbl_nocl].x.values, tsne_result[lbl_nocl].y.values, c="white")
    scatter = plt.scatter(tsne_result[lbl_wicl].x.values, tsne_result[lbl_wicl].y.values, c=label_ids[lbl_wicl], cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)
    plt.tight_layout()

    plt.show()


def show_tsne_only_selected_columns(query_matcher: QueryMatcher, idx, perplexity, n_iter, lr, col_selector=".*"):
    flat_data = query_matcher.features_df_properly_scaled.filter(regex=col_selector, axis=1)
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    # ax = fig.subplot()

    tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("nipy_spectral", len(unique_label_ids))
    scatter = plt.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)
    plt.tight_layout()

    plt.show()


def show_tsne_PCA(query_matcher, idx, perplexity, n_iter, lr):
    flat_data = query_matcher.features_df_properly_scaled
    flat_data = pd.DataFrame(PCA(n_components=50).fit_transform(flat_data), index=flat_data.index)
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    # ax = fig.subplot()

    tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("nipy_spectral", len(unique_label_ids))
    scatter = plt.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)
    plt.tight_layout()

    plt.show()


def run_parallel_iteration(params):
    query_matcher, idx, perplexity, n_iter, lr = params
    identifier = constuct_identifier("tsne", perplexity, n_iter, lr)
    flat_data = query_matcher.features_df_properly_scaled
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    # ax = fig.subplot()

    tsne_result = compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("hsv", len(unique_label_ids))
    scatter = plt.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    plt.tight_layout()
    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)

    plt.savefig(f"trash/tsne/{identifier}.png")
    plt.close()

    print("Done ", idx)


def run_parallel_iteration_3D(params):
    query_matcher, idx, perplexity, n_iter, lr = params
    identifier = constuct_identifier("tsne", perplexity, n_iter, lr)
    flat_data = query_matcher.features_df_properly_scaled
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    tsne_result = compute_tsne_3D(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("hsv", len(unique_label_ids))
    scatter = ax.scatter(tsne_result.x.values, tsne_result.y.values, tsne_result.z.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    plt.tight_layout()
    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)

    plt.savefig(f"trash/tsne_3D/{identifier}.png")
    plt.close()

    print("Done ", idx)


def run_parallel_iteration_scaled(params):
    query_matcher, idx, perplexity, n_iter, lr, col_selector = params
    identifier = constuct_identifier(f"tsne_{idx.replace('.*', 'all')}", perplexity, n_iter, lr)
    file_name = f"trash/tsne_2D_scaled/{identifier}.png"
    if os.path.exists(file_name):
        return None
    flat_data = query_matcher.features_df_all_scaled.filter(regex=col_selector, axis=1)
    names_n_labels = [[name, query_matcher.map_to_label(name, True)] for name in list(flat_data.index)]
    classification_indexes = {label: idx for idx, label in enumerate(sorted(list(set([label for _, label in names_n_labels]))))}
    unique_label_ids = list(classification_indexes.values())
    unique_labels = list(classification_indexes.keys())
    figsize = (10, 8)
    fig = plt.figure(idx, figsize=figsize)
    ax = fig.add_subplot(111)

    tsne_result = compute_tsne_3D(flat_data, perplexity, n_iter, lr, names_n_labels, True)
    label_ids = [classification_indexes[label] for label in tsne_result.label]
    cmap = plt.cm.get_cmap("hsv", len(unique_label_ids))
    scatter = ax.scatter(tsne_result.x.values, tsne_result.y.values, c=label_ids, cmap=cmap)
    plt.title(f"Perplexity ({perplexity}), Iterations ({n_iter}), LR ({lr})")

    plt.tight_layout()
    cbar = plt.colorbar(scatter, ticks=unique_label_ids)
    cbar.ax.set_yticklabels(unique_labels)

    plt.savefig(file_name)
    plt.close()

    print("Done ", identifier)


def run_parallel_exploration(query_matcher):
    perplexity_range = np.arange(1, 1800)
    iterations_range = np.arange(250, 10000, 10)
    learning_rates_range = np.logspace(0, 4, base=10, num=100)

    combinations = []
    for idx in range(1000):
        pr = np.random.choice(perplexity_range)
        ir = np.random.choice(iterations_range)
        lrr = np.random.choice(learning_rates_range)
        combinations.append((query_matcher, idx, pr, ir, lrr))

    pool = mp.Pool(6)
    results = pool.imap(run_parallel_iteration, tqdm(combinations, total=len(combinations)))
    gather = list(results)


def run_parallel_exploration_3D(query_matcher):
    perplexity_range = np.linspace(1, 500, 250)
    iterations_range = [10000]
    learning_rates_range = np.linspace(1, 250, 100)

    combinations = []
    for idx in range(1000):
        pr = np.random.choice(perplexity_range)
        ir = np.random.choice(iterations_range)
        lrr = np.random.choice(learning_rates_range)
        combinations.append((query_matcher, idx, pr, ir, lrr))

    pool = mp.Pool(9)
    results = pool.imap(run_parallel_iteration_3D, tqdm(set(combinations), total=len(combinations)))
    gather = list(results)


def run_parallel_exploration_scaled(query_matcher):
    perplexity_range = np.linspace(1, 500, 250)
    iterations_range = [5000]
    learning_rates_range = np.linspace(1, 250, 100)

    combinations = []
    for idx in range(1000):
        selection = np.random.choice(["scalar_", "hist_", "skeleton_", ".*"])
        pr = np.random.choice(perplexity_range)
        ir = np.random.choice(iterations_range)
        lrr = np.random.choice(learning_rates_range)
        combinations.append((query_matcher, selection, pr, ir, lrr, f"^{selection}"))

    pool = mp.Pool(6)
    results = pool.imap(run_parallel_iteration_scaled, tqdm(set(combinations), total=len(combinations)))
    gather = list(results)
    # run_parallel_iteration_scaled(combinations[0])