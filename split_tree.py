# %%
import numpy as np
import pyvista as pv
from pyvista import examples
from itertools import product
# %% Load mesh
mesh = pv.PolyData(examples.download_cow().triangulate().points[:10])

# %%


class Node(object):
    # children = []

    def __init__(self, parent, points):
        self.parent = parent
        self.points = points
        self.bounds = pv.PolyData(points).bounds
        self.diameter = np.sqrt(np.sum(np.diff(np.array(self.bounds).reshape((-1, 2)), axis=1)))
        self.node_l = None
        self.node_r = None

    def append_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node: {len(self.points)} points and diameter ({self.diameter})"

    def _split(self):
        # print(len(self.points))
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

        # self.children.append(Node(self, v.points))
        self.node_l = Node(self, u.points)._split()
        self.node_r = Node(self, v.points)._split()
        # self.append_child(lc)
        # self.append_child(rc)
        return self


class Tree(object):
    def __init__(self, root_node):
        self.root = root_node

    def split_tree(self):
        split_points = self.root._split()
        # print(split_points)

    def find_pairs(self):
        self.pairs = self._find_pairs(self.root, self.root)

    @staticmethod
    def _find_pairs(u, v, eps=1):
        if u == v and u.diameter == 0:
            return None
        if u.diameter < v.diameter:
            tmp = u
            u = v
            v = tmp
        if u.diameter <= eps * Tree._distance(u, v):
            return (u, v)
        return (Tree._find_pairs(u.node_l, v), Tree._find_pairs(u.node_r, v))

    @staticmethod
    def _distance(u, v):
        vertices1, vertices2 = list(zip(*product(u.points, v.points)))
        difference_between_points = np.array(vertices1) - np.array(vertices2)
        squared_difference = np.square(difference_between_points)
        sum_of_squared = np.sum(squared_difference, axis=1)
        L2_distance = np.sqrt(sum_of_squared)
        min_distance = np.min(L2_distance)
        return min_distance

    @staticmethod
    def traverse(t, level=0, indent=4):
        if not t:
            return "LEAF"
        value = t[0].__repr__()
        if level > 0:
            prefixed_str = ' ' * (indent * (level - 1)) + '+---'
        else:
            prefixed_str = ''
        result_str = prefixed_str + value
        for child in t[1:]:
            result_str += Tree.traverse(child, level + 1)
        return result_str

    def __repr__(self):
        return Tree.traverse(self.pairs)


class Leaf(object):
    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"Leaf: {len(self.points)} points"


tree = Tree(Node(None, mesh.points))
tree.split_tree()
tree.find_pairs()
print(tree)
# %%

# %%
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
