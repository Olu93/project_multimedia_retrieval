# %%
from pyvista import examples
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing as mp
import math
from itertools import product
from helper.diameter_computer import compute_comparison
from tqdm import tqdm
import pandas as pd


def run_diameter_experiment(data):
    print(data)
    num_experiments = 10
    num_iterations = 10
    pool = mp.Pool(math.ceil(mp.cpu_count() * .75))
    params = list(product((data.points, ), np.linspace(len(data.points) / 10, len(data.points), num_experiments), range(num_iterations)))
    experiment_results = pool.map(compute_comparison, tqdm(params, total=len(params)))

    return pd.DataFrame(list(experiment_results))


# %%
if __name__ == "__main__":
    experiment_results = run_diameter_experiment(examples.download_bunny().triangulate())
    experiment_results

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharex=True)
    x_axis = experiment_results.iloc[:, 0]
    diameter_plot = axes[0]
    diameter_plot.plot(x_axis, experiment_results.iloc[:, 1], label="approx")
    diameter_plot.plot(x_axis, experiment_results.iloc[:, 2], label="exact")
    diameter_plot.set_title("Diameter")
    diameter_plot.legend()
    computation_time_plot = axes[1]
    computation_time_plot.plot(x_axis, experiment_results.iloc[:, 3], label="approx")
    computation_time_plot.plot(x_axis, experiment_results.iloc[:, 4], label="exact")
    computation_time_plot.set_title("Computation time (in sec)")
    computation_time_plot.legend()
    plt.tight_layout()
    plt.show()