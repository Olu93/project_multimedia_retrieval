# %%
import numpy as np
import pyvista as pv
from pyvista import examples
from itertools import product
import treelib
import heapq
import random

# %% Load mesh
mesh = pv.PolyData(examples.download_cow().triangulate().points[:500])

# %%


class Node(object):
    # children = []

    def __init__(self, parent, points):
        self.parent = parent
        self.points = points
        poly_data = pv.PolyData(points)
        self.bounds = poly_data.bounds
        self.center = np.array(poly_data.center)
        self.bbox_lengths = np.diff(np.array(self.bounds).reshape((-1, 2)), axis=1)
        self.bbox_diameter = np.sqrt(np.sum(self.bbox_lengths))
        self.bsphere_radius = self.bbox_lengths.max() / 2
        self.node_l = None
        self.node_r = None

    def append_child(self, node):
        self.children.append(node)

    def __repr__(self):
        return f"Node: p:{len(self.points)} - d:{self.diameter})"

    def display(self):
        lines, *_ = self._display_aux()
        return "\n".join(lines)

    def _display_aux(self):
        """Returns list of strings, width, height, and horizontal coordinate of the root."""
        # No child.
        if self.node_r is None and self.node_l is None:
            line = '%s' % len(self.points)
            width = len(line)
            height = 1
            middle = width // 2
            return [line], width, height, middle

        # Only left child.
        if self.node_r is None:
            lines, n, p, x = self.node_l._display_aux()
            s = '%s' % len(self.points)
            u = len(s)
            first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s
            second_line = x * ' ' + '/' + (n - x - 1 + u) * ' '
            shifted_lines = [line + u * ' ' for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, n + u // 2

        # Only right child.
        if self.node_l is None:
            lines, n, p, x = self.node_r._display_aux()
            s = '%s' % len(self.points)
            u = len(s)
            first_line = s + x * '_' + (n - x) * ' '
            second_line = (u + x) * ' ' + '\\' + (n - x - 1) * ' '
            shifted_lines = [u * ' ' + line for line in lines]
            return [first_line, second_line] + shifted_lines, n + u, p + 2, u // 2

        # Two children.
        left, n, p, x = self.node_l._display_aux()
        right, m, q, y = self.node_r._display_aux()
        s = '%s' % len(self.points)
        u = len(s)
        first_line = (x + 1) * ' ' + (n - x - 1) * '_' + s + y * '_' + (m - y) * ' '
        second_line = x * ' ' + '/' + (n - x - 1 + u + y) * ' ' + '\\' + (m - y - 1) * ' '
        if p < q:
            left += [n * ' '] * (q - p)
        elif q < p:
            right += [m * ' '] * (p - q)
        zipped_lines = zip(left, right)
        lines = [first_line, second_line] + [a + u * ' ' + b for a, b in zipped_lines]
        return lines, n + m + u, max(p, q) + 2, n + u // 2

    def _split(self):
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

        self.node_l = Node(self, u.points)._split()
        self.node_r = Node(self, v.points)._split()

        return self


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
        return self.M < other.M

    def __repr__(self):
        return f"(u:{self.u_num_points}, v:{self.v_num_points}, {self.M})"


class Tree(object):
    def __init__(self, root_node):
        self.root = root_node

    def split_tree(self):
        split_points = self.root._split()
        # print(split_points)

    def find_pairs(self):
        self.pairs = self._find_pairs(self.root, self.root)

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
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u, v.node_r)
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u, v.node_l)
            if pair.v_num_points == 1:
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v)
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v)
            if pair.u_num_points > 1 and pair.v_num_points > 1:
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v.node_l)
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v.node_r)
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_l, v.node_r)
                if u == v:
                    continue
                p_curr, delta_curr = Tree.add_to_heap(p_curr, curr_limit, delta_curr, u.node_r, v.node_l)
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
    def _find_diam(u, v, eps=.5):
        if u == v and u.diameter == 0:
            return None
        if u.bbox_diameter < v.bbox_diameter:
            tmp = u
            u = v
            v = tmp
        if u.bbox_diameter <= eps * Tree._distance(u, v):
            return (u, v)
        return (Tree._find_pairs(u.node_l, v), Tree._find_pairs(u.node_r, v))

    @staticmethod
    def _find_pairs(u, v, eps=.5):
        if u == v and u.bbox_diameter == 0:
            return None
        if u.bbox_diameter < v.bbox_diameter:
            tmp = u
            u = v
            v = tmp
        if u.bbox_diameter <= eps * Tree._distance(u, v):
            return (u, v)
        return (Tree._find_pairs(u.node_l, v), Tree._find_pairs(u.node_r, v))

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
        return Tree.traverse(self.pairs)


class Leaf(object):
    def __init__(self, points):
        self.points = points

    def __repr__(self):
        return f"Leaf: {len(self.points)} points"


tree = Tree(Node(None, mesh.points))
tree.split_tree()
tree.find_pairs()
# print(tree.root.display())

# %%
from pprint import pprint
# pprint(tree.pairs, indent=4)
print(tree.find_diam())
print(Tree._distance(tree.root, tree.root))
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
