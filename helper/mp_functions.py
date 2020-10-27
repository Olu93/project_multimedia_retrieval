import jsonlines
import numpy as np
import multiprocess as mp
import math
import itertools
from tqdm import tqdm
import random
import pyvista as pv
from .diameter_computer import Node, AprxDiamWSPDRecursive
import pandas as pd
from itertools import product


# https://stackoverflow.com/a/13530258/4162265
def compute_feature_extraction(extractor, data):
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    extractions = pool.imap(extractor.mono_run_pipeline, data, chunksize=1)
    return extractions


def compute_normalization(normalizer, data):
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    normalized = pool.imap_unordered(normalizer.mono_run_pipeline, data, chunksize=10)
    return normalized


def compute_read(reader, data):
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    read = pool.imap_unordered(reader.mono_run_pipeline, data, chunksize=5)
    return read


def compute_distance(points):
    all_combinations = itertools.product(points, points)
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    calculations = pool.imap_unordered(point_distance, all_combinations, chunksize=1000000)
    return np.max(list(tqdm(calculations, total=len(points)**2)))


def point_distance(points):
    p1, p2 = points
    return np.linalg.norm(p1 - p2)


def listener(args):
    '''listens for messages on the q, writes to file. '''
    print("SPINNING UP THE QUEUE!!!")
    target_file, q = args
    with jsonlines.open(target_file, "w", flush=True) as writer:
        while 1:
            m = q.get()
            if m == 'kill':
                break
            writer.write(m)
            print(f"Write {m['name']} into file!")
