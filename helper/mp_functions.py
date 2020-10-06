import numpy as np
import multiprocessing as mp
import math
import itertools
from tqdm import tqdm


def compute_feature_extraction(extractor, data):
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    extractions = pool.imap_unordered(extractor.mono_run_pipeline, data, chunksize=10)
    return extractions


def compute_normalization(normalizer, data):
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    normalized = pool.imap_unordered(normalizer.mono_run_pipeline, data, chunksize=10)
    return normalized


def compute_distance(points):
    all_combinations = itertools.product(points, points)
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    calculations = pool.imap_unordered(point_distance, all_combinations, chunksize=1000000)
    return np.max(list(tqdm(calculations, total=len(points)**2)))


def point_distance(points):
    p1, p2 = points
    return np.linalg.norm(p1 - p2)
