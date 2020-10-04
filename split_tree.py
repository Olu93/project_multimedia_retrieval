# %%
import numpy as np
import pyvista as pv
from pyvista import examples
from itertools import product
import treelib
import heapq
import random
from anytree import NodeMixin, RenderTree
# %% Load mesh
mesh = pv.PolyData(examples.download_cow().triangulate().points[:50])

# %%


class Node(NodeMixin):
    # children = []

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
        # self.node_l = None
        # self.node_r = None

    def get_children(self):
        return self.children[0], self.children[1]

    def __repr__(self):
        return self.name

    def split_fair(self):
        if len(self.points) <= 1:
            return self
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


fair_split_tree = Node(mesh.points, parent=None).split_fair()
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

    @staticmethod
    def M(u, v):
        return np.linalg.norm(u.center - v.center) + u.bsphere_radius + v.bsphere_radius

    def get_pair(self):
        return self.u, self.v

    # Shout out to: https://stackoverflow.com/a/1227152/4162265
    def __eq__(self, other):
        return self.M == other.M

    def __lt__(self, other):
        # Switched sign in order to make it work with min-heap assumption of the heapq package
        return self.M > other.M

    def __repr__(self):
        return f"(u:{self.u_num_points}, v:{self.v_num_points}, {self.M})"


class FairSplitTree(object):
    def __init__(self, root_node):
        self.root = root_node

    def split_tree(self):
        self.root._split()

    def find_diam(self, eps=.01):
        p_curr = []
        delta_curr = self.root.bbox_diameter
        starting_point = Pair(self.root, self.root)
        heapq.heapify(p_curr)
        heapq.heappush(p_curr, starting_point)
        while p_curr:
            pair = heapq.heappop(p_curr)
            m = pair.M
            u, v = pair.get_pair()
            curr_limit = (1 + eps) * delta_curr
            if pair.u_num_points < 1 or pair.v_num_points < 1 or m <= curr_limit:
                continue
            if pair.u_num_points == 1:
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u, v.node_r)
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u, v.node_l)
            if pair.v_num_points == 1:
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v)
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v)
            if pair.u_num_points > 1 and pair.v_num_points > 1:
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v.node_l)
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v.node_r)
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v.node_r)
                if u == v:
                    continue
                p_curr, delta_curr = FairSplitTree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v.node_l)
        return delta_curr

    @staticmethod
    def add_to_heap(heap, limit, delta_curr, u, v):
        pair = Pair(u, v)
        m = pair.M
        if m <= limit:
            return heap, delta_curr
        L2_distance = np.linalg.norm(random.choice(u.points) - random.choice(v.points))
        delta_curr = L2_distance if L2_distance > delta_curr else delta_curr
        heapq.heappush(heap, pair)
        return heap, delta_curr

    @staticmethod
    def _distance(u, v):
        vertices1, vertices2 = list(zip(*product(u.points, v.points)))
        difference_between_points = np.array(vertices1) - np.array(vertices2)
        squared_difference = np.square(difference_between_points)
        sum_of_squared = np.sum(squared_difference, axis=1)
        L2_distance = np.sqrt(sum_of_squared)
        max_distance = np.max(L2_distance)
        return max_distance

    def __repr__(self):
        return FairSplitTree.traverse(self.pairs)


class Leaf(object):
    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"Leaf: {len(self.points)} points"


tree = FairSplitTree(Node(None, mesh.points))
tree.split_tree()
tree.find_pairs()
# print(tree.root.display())

# %%
from pprint import pprint
# pprint(tree.pairs, indent=4)
print(tree.find_diam())
print(FairSplitTree._distance(tree.root, tree.root))
# %%
# dataset = examples.download_bunny_coarse()
# clipped = dataset.clip((-1, 0, 0))

# p = pv.Plotter()
# p.add_mesh(dataset, style='wireframe', color='blue', label='Input')
# p.add_mesh(clipped, label='Clipped')
# p.add_legend()
# p.add_bounds_axes()
# p.camera_position = [(0.24, 0.32, 0.7), (0.02, 0.03, -0.02), (-0.12, 0.93, -0.34)]
# p.show()

# %%
