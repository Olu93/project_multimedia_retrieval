import heapq
import random
import time
from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pyvista as pv
from anytree import NodeMixin, RenderTree
from pyvista import examples
from tqdm import tqdm
from helper.diameter_computer import Pair, AprxDiameter, Node


def compute_comparison(data, num_points):
    mesh = pv.PolyData(random.sample(list(data.points), num_points))
    root = Node(mesh.points, parent=None)
    diameter_computer = AprxDiamWSPDRecursive(root)
    diameter_computer.compute_approx_diameter()
    diameter_computer.compute_exact_diameter()
    print(f"Finished {num_points}: exact({diameter_computer.exact_diameter}) - approx({diameter_computer.approx_diameter})")
    return {
        "Number of points": num_points,
        "Approximated diameter": diameter_computer.approx_diameter,
        "Exact diameter": diameter_computer.exact_diameter,
        "Approx. diameter computation time (in sec)": diameter_computer.approx_time,
        "Exact. diameter computation time (in sec)": diameter_computer.exact_time,
        "Approx. diameter computation time (in sec) [Without tree constiruction]": diameter_computer.approx_time_only_algorithm,
    }


class AprxDiamWSPDRecursive(AprxDiameter):
    def compute_approx_diameter(self, eps=.01):
        self.eps = eps
        # print(" Starting approximation")
        start_time = time.time()
        start_time_only_algorithm = time.time()
        starting_pair = Pair(self.root, self.root)
        u_root_repr, v_root_repr = starting_pair.get_pair_reprensetantives()
        delta_curr = np.linalg.norm(u_root_repr - v_root_repr)
        # points_curr = self.root.min_max_points
        points_curr = (self.root.center, self.root.center)
        delta_curr, points_curr = self.split(starting_pair, delta_curr, points_curr)
        self.approx_points = points_curr
        self.approx_diameter = delta_curr
        self.approx_time = time.time() - start_time
        self.approx_time_only_algorithm = time.time() - start_time_only_algorithm
        return delta_curr

    def split(self, pair, delta_curr, points_curr):
        # print(pair)
        points_curr = pair.get_pair_reprensetantives()
        u_representative, v_representative = points_curr
        initial_candidate = (delta_curr, points_curr)
        m = pair.M
        L2_distance = np.linalg.norm(u_representative - v_representative)
        delta_curr = L2_distance if L2_distance > delta_curr else delta_curr
        curr_limit = (1 + self.eps) * delta_curr

        if pair.u_num_points < 1 or pair.v_num_points < 1 or m <= curr_limit:
            return (delta_curr, points_curr)
        if pair.u_num_points == 1 or pair.v_num_points == 1:
            keep, left, right = self._one_sided_split(pair)
        if pair.u_num_points > 1 and pair.v_num_points > 1:
            larger_node, keep = pair.get_split_candidate_and_non_split_candidate()
            left, right = Node._split_fair(larger_node).get_children()

        results_1, results_2 = self.next_traversal(keep, left, right, delta_curr, points_curr)
        candidates = (initial_candidate, results_1, results_2)
        max_idx = np.argmax([item[0] for item in candidates])
        return candidates[max_idx]

    def next_traversal(self, keep, left, right, delta_curr, points_curr):
        pair_1 = Pair(keep, left)
        pair_2 = Pair(keep, right)
        curr_limit = (1 + self.eps) * delta_curr
        results_1 = (delta_curr, points_curr) if pair_1.M <= curr_limit else self.split(pair_1, delta_curr, points_curr)
        delta_curr, points_curr = results_1 if results_1[0] > delta_curr else (delta_curr, points_curr)
        results_2 = (delta_curr, points_curr) if pair_2.M <= curr_limit else self.split(pair_2, delta_curr, points_curr)
        return results_1, results_2

    def _one_sided_split(self, pair):
        keep, left, right = None, None, None
        if pair.u_num_points == 1:
            u, v = pair.get_pair()
            left, right = Node._split_fair(v).get_children()
            keep = u
        if pair.v_num_points == 1:
            u, v = pair.get_pair()
            left, right = Node._split_fair(u).get_children()
            keep = v
        return keep, left, right

    def compute_exact_diameter(self):
        # print("Starting exact computation")
        return super().compute_exact_diameter()


data = examples.download_bunny().triangulate().decimate(.7)
mesh = pv.PolyData(data.points[:50])
root = Node(mesh.points, parent=None)
diameter_computer = AprxDiamWSPDRecursive(root)
print(diameter_computer.compute_approx_diameter())
print(diameter_computer.compute_exact_diameter())
# diameter_computer.show()

num_experiments = 10
experiment_results = pd.DataFrame(
    [compute_comparison(data, int(num_points)) for num_points in tqdm(np.linspace(len(data.points) / 10, len(data.points), num_experiments), total=num_experiments)])
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