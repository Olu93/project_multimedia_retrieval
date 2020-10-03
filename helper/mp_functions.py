import numpy as np


def compute_distance(points):
    p1, p2 = points
    return np.linalg.norm(p1 - p2)
