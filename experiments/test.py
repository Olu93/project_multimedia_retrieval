
from experiments.experiment_TSNE import run_parallel_iteration
from query_matcher import QueryMatcher
import numpy as np
import multiprocessing as mp
from helper.config import FEATURE_DATA_FILE


if __name__ == "__main__":
    perplexity_range = np.arange(1, 1800)
    iterations_range = np.arange(0, 10000, 11)
    learning_rates_range = np.logspace(1, 1000, 10)
    query_matcher = QueryMatcher(FEATURE_DATA_FILE)
    combinations = []
    for idx in range(1000):
        pr = np.random.choice(perplexity_range)
        ir = np.random.choice(iterations_range)
        lrr = np.random.choice(learning_rates_range)
        combinations.append((query_matcher, idx, pr, ir, lrr))

    pool = mp.Pool(2)
    pool.map(run_parallel_iteration, combinations)