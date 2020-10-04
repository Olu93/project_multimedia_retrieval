# %%
import numpy as np
import pyvista as pv
from pyvista import examples
from itertools import product
import treelib
import heapq
import random
from anytree import NodeMixin, RenderTree
import time
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt

# %% Load mesh
data = examples.download_cow().triangulate()

# %%


class Node(NodeMixin):
    def __init__(self, points, parent=None, children=None):
        super(Node, self).__init__()
        self.parent = parent
        self.points = points
        poly_data = pv.PolyData(points)
        self.bounds = poly_data.bounds
        self.center = np.array(poly_data.center)
        self.bbox_lengths = np.diff(np.array(self.bounds).reshape((-1, 2)), axis=1)
        self.bbox_diameter = np.sqrt(np.sum(self.bbox_lengths))
        self.bsphere_radius = self.bbox_lengths.max() / 2
        self.name = f"Node ({len(points)} points, radius:{self.bsphere_radius:.2f})"

    def get_children(self):
        # Returns the left and right child of the node
        return self.children[0], self.children[1]

    def __repr__(self):
        return self.name

    def split_fair(self):
        if len(self.points) <= 1:
            return Leaf(self.points)
        differences = np.abs(np.diff(np.array(self.bounds).reshape((-1, 2)), axis=1))
        longest_axis = np.argmax(differences)
        splitting_plane_normal = np.zeros_like(differences)
        splitting_plane_normal[longest_axis] = 1
        splitting_plane_normal = splitting_plane_normal.flatten()
        dataset = pv.PolyData(self.points)
        u = dataset.clip(splitting_plane_normal)
        v = dataset.clip(-1 * splitting_plane_normal)
        self.children = (Node(u.points, parent=self).split_fair(), Node(v.points, parent=self).split_fair())
        return self

    @staticmethod
    def show_tree(root_node):
        for pre, fill, node in RenderTree(root_node):
            print("%s%s" % (pre, node.name))


class Leaf(Node):
    def __init__(self, points):
        self.points = points
        self.name = f"Leaf: ({len(self.points)} points)"
        self.bsphere_radius = 0
        self.center = points[0]

    def get_children(self):
        # Returns the left and right child of the node
        return None, None


mesh = pv.PolyData(data.points[:50])
fair_split_tree = Node(mesh.points).split_fair()
for pre, fill, node in RenderTree(fair_split_tree):
    print("%s%s" % (pre, node.name))


# %%
class Pair(object):
    """
    Comparable pair of nodes in the tree. 
    """
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.M = Pair.M(u, v)
        self.u_num_points = len(u.points)
        self.v_num_points = len(v.points)
        self.u_representative = random.choice(u.points)
        self.v_representative = random.choice(v.points)

    @staticmethod
    def M(u, v):
        return np.linalg.norm(u.center - v.center) + u.bsphere_radius + v.bsphere_radius

    def get_pair(self):
        return self.u, self.v

    def get_pair_reprensetantives(self):
        return self.u_representative, self.v_representative

    # Shout out to: https://stackoverflow.com/a/1227152/4162265
    def __eq__(self, other):
        return self.M == other.M

    def __lt__(self, other):
        # Switched sign in order to make it work with min-heap assumption of the heapq package
        return self.M > other.M

    def __repr__(self):
        return f"(u:{self.u_num_points}, v:{self.v_num_points}, {self.M})"


class AprxDiameter(object):
    def __init__(self, root_node):
        self.root = root_node.split_fair()

    def compute_approx_diameter(self, eps=.01):
        start_time = time.time()
        p_curr = []
        delta_curr = self.root.bbox_diameter
        starting_pair = Pair(self.root, self.root)
        points_curr = starting_pair.get_pair_reprensetantives()
        heapq.heapify(p_curr)
        heapq.heappush(p_curr, starting_pair)
        while p_curr:
            pair = heapq.heappop(p_curr)
            m = pair.M
            u, v = pair.get_pair()
            u_left, u_right = u.get_children()
            v_left, v_right = v.get_children()
            curr_limit = (1 + eps) * delta_curr
            if pair.u_num_points < 1 or pair.v_num_points < 1 or m <= curr_limit:
                continue
            if pair.u_num_points == 1:
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u, v_right)
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u, v_left)
            if pair.v_num_points == 1:
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_right, v)
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_left, v)
            if pair.u_num_points > 1 and pair.v_num_points > 1:
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_left, v_left)
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_right, v_right)
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_left, v_right)
                if u == v:
                    continue
                p_curr, delta_curr, points_curr = AprxDiameter.add_to_heap(p_curr, curr_limit, delta_curr, u_right, v_left)

        self.approx_points = points_curr
        self.approx_diameter = delta_curr
        self.approx_time = time.time() - start_time
        return delta_curr

    @staticmethod
    def add_to_heap(heap, limit, delta_curr, u, v):
        pair = Pair(u, v)
        points_curr = pair.get_pair_reprensetantives()
        u_representative, v_representative = points_curr
        m = pair.M
        if m <= limit:
            return heap, delta_curr, points_curr

        L2_distance = np.linalg.norm(u_representative - v_representative)
        points_curr = pair.get_pair_reprensetantives()
        delta_curr = L2_distance if L2_distance > delta_curr else delta_curr
        heapq.heappush(heap, pair)
        return heap, delta_curr, points_curr

    def compute_exact_diameter(self):
        start_time = time.time()
        max_distance, exact_points = self._exact_diameter(self.root, self.root)
        self.exact_points = exact_points
        self.exact_diameter = max_distance
        self.exact_time = time.time() - start_time
        return self.exact_diameter

    @staticmethod
    def _exact_diameter(u, v):
        vertices1, vertices2 = list(zip(*product(u.points, v.points)))
        difference_between_points = np.array(vertices1) - np.array(vertices2)
        squared_difference = np.square(difference_between_points)
        sum_of_squared = np.sum(squared_difference, axis=1)
        L2_distance = np.sqrt(sum_of_squared)
        max_distance = np.max(L2_distance)
        max_distance_idx = np.argmax(L2_distance)
        exact_points = (vertices1[max_distance_idx], vertices2[max_distance_idx])
        return max_distance, exact_points

    def show(self, **kwargs):
        p = pv.Plotter()
        mesh = pv.PolyData(self.root.points)
        p.add_mesh(mesh, color='blue', **kwargs)
        if self.approx_diameter:
            a, b = self.approx_points
            line = pv.Line(a, b)
            p.add_mesh(line, color='red', label=f'approx distance ({self.approx_diameter:.4f})')
        if self.exact_diameter:
            a, b = self.exact_points
            line = pv.Line(a, b)
            p.add_mesh(line, color='green', label=f'exact distance ({self.exact_diameter:.4f})')

        p.add_legend()
        p.show()

    def __repr__(self):
        return AprxDiameter.traverse(self.pairs)


mesh = pv.PolyData(data.points[:1000])
root = Node(mesh.points, parent=None)
diameter_computer = AprxDiameter(root.split_fair())
print(diameter_computer.compute_approx_diameter())
print(diameter_computer.compute_exact_diameter())

# %%
diameter_computer.show()


# %%
def compute_comparison(data, num_points):

    mesh = pv.PolyData(data.points[:num_points])
    root = Node(mesh.points, parent=None)
    diameter_computer = AprxDiameter(root.split_fair())
    diameter_computer.compute_approx_diameter()
    diameter_computer.compute_exact_diameter()
    return {
        "Number of points": num_points,
        "Approximated diameter": diameter_computer.approx_diameter,
        "Exact diameter": diameter_computer.exact_diameter,
        "Approx. diameter computation time (in sec)": diameter_computer.approx_time,
        "Exact. diameter computation time (in sec)": diameter_computer.exact_time,
    }


num_experiments = 10
experiment_results = pd.DataFrame([compute_comparison(data, int(num_points)) for num_points in tqdm(np.linspace(50, len(data.points), num_experiments), total=num_experiments)])
experiment_results

# %%
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