from datetime import datetime
from helper.misc import jsonify
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
from pprint import pprint
import time
# https://stackoverflow.com/a/13530258/4162265
def compute_feature_extraction(extractor, data):
    manager = mp.Manager()
    q = manager.Queue()
    process_list = []
    for item in data:
        q.put(item)
    # progress_bar = tqdm(total=len(data))
    for i in range(math.ceil(mp.cpu_count() * .75)):
        p = mp.Process(target=listener, args=(extractor, q, i))
        process_list.append(p)

    p = mp.Process(target=heart_beat_check, args=(process_list,))
    for proc in process_list + [p]:
        proc.start()

    for p in process_list:
        p.join()

    return True


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


def listener(extractor, q, index):
    '''listens for messages on the q, writes to file. '''
    print("SPINNING UP THE PROCESS!!!")
    timestamp = str(datetime.date(datetime.now()))
    file_name = f"stats/tmp/tmp-{index}.jsonl"

    with jsonlines.open(file_name, mode="w", flush=True) as writer:
        while not q.empty():
            m = q.get(block=True)
            result = extractor.mono_run_pipeline(m)
            result = jsonify(result)
            result = dict(timestamp=timestamp, **result)
            # pprint({key: type(val) if not type(val) == list else list(set([type(num) for num in val])) for key, val in result.items()})
            writer.write(result)
            print(f"Write {result['name']} into file {file_name}!")
            # tqdm.update(1)


def heart_beat_check(process_list):
    num_alive = len(process_list)
    print(f"Spinning up heart beat check - {num_alive} hearts - Badumm Tzzz....")
    while num_alive == 0:
        num_alive = sum([p.is_alive() for p in process_list])
        print(f"{num_alive} processes sill alive and kicking!")
        time.sleep(1)
