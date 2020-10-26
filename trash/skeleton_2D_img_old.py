# %%
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.morphology.binary import binary_closing
from skimage.util import dtype
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, skeletonize_3d
from skimage.util import invert
from scipy.spatial import ConvexHull, Delaunay, Voronoi
from scipy.spatial.qhull import QhullError
import skimage
import sknw
import networkx as nx
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")
mesh.plot()

def prepare_image(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


def extract_sillhouettes(mesh):
    images = []
    normal = np.zeros((3, 1))
    p = pv.Plotter()
    for i in range(3):
        normal[:] = 0
        normal[i] = -1
        # cpos = [normal * 2, (0, 0, 0), (0, 0, 1.0)]
        projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
        p.add_mesh(projected)
        p.set_position(normal * 2)
        images.append(prepare_image(p.get_image_depth()))
    return images


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


sillhouettes = extract_sillhouettes(mesh)
skeletons = extract_skeletons(sillhouettes)
graphs = extract_graphs(skeletons)
extracted_information = list(zip(sillhouettes, skeletons, graphs))
# %%p.plot
fig = plt.figure(figsize=(12, 10))
for idx, (img_array, skeleton) in enumerate(zip(sillhouettes, skeletons)):
    ax = fig.add_subplot(2, 3, idx + 1)
    ax.imshow(img_array, cmap=plt.cm.gray)
    ax = fig.add_subplot(2, 3, 3 + idx + 1)
    ax.imshow(skeleton, cmap=plt.cm.gray)
plt.tight_layout()
plt.show()

# %%

# %%
import cv2


def skeleton_endpoints(skel):
    # https://stackoverflow.com/a/38867556/4162265
    # make out input nice, possibly necessary
    skel = skel.copy()
    skel[skel != 0] = 1
    skel = np.uint8(skel)

    # apply the convolution
    kernel = np.uint8([[1, 1, 1], [1, 10, 1], [1, 1, 1]])
    src_depth = -1
    filtered = cv2.filter2D(skel, src_depth, kernel)

    # now look through to find the value of 11
    # this returns a mask of the endpoints, but if you just want the coordinates, you could simply return np.where(filtered==11)
    out = np.zeros_like(skel)
    pos = np.where(filtered == 11)
    out[pos] = 1
    return out, np.array(pos).T



def highlight_edges(skel, func, scale=3):
    map_points, positions = func(skel)
    for pos in positions:
        x, y = pos
        for i in range(1, scale + 1):
            map_points[x + i, y + i] = 1
            map_points[x - i, y + i] = 1
            map_points[x + i, y - i] = 1
            map_points[x - i, y - i] = 1
    return map_points


which = 2
plt.imshow(skeletons[which] + highlight_edges(skeletons[which], skeleton_endpoints, scale=10), cmap=plt.cm.gray)
# %%
# Functions to generate kernels of curve intersection
# https://stackoverflow.com/a/54112617/4162265
import itertools
def generate_nonadjacent_combination(input_list, take_n):
    """ 
    It generates combinations of m taken n at a time where there is no adjacent n.
    INPUT:
        input_list = (iterable) List of elements you want to extract the combination 
        take_n =     (integer) Number of elements that you are going to take at a time in
                        each combination
    OUTPUT:
        all_comb =   (np.array) with all the combinations
    """
    all_comb = []
    for comb in itertools.combinations(input_list, take_n):
        comb = np.array(comb)
        d = np.diff(comb)
        fd = np.diff(np.flip(comb))
        if len(d[d == 1]) == 0 and comb[-1] - comb[0] != 7:
            all_comb.append(comb)
    return all_comb


def populate_intersection_kernel(combinations):
    """
    Maps the numbers from 0-7 into the 8 pixels surrounding the center pixel in
    a 9 x 9 matrix clockwisely i.e. up_pixel = 0, right_pixel = 2, etc. And 
    generates a kernel that represents a line intersection, where the center 
    pixel is occupied and 3 or 4 pixels of the border are ocuppied too.
    INPUT:
        combinations = (np.array) matrix where every row is a vector of combinations
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    n = len(combinations[0])
    template = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")
    match = [(0, 1), (0, 2), (1, 2), (2, 2), (2, 1), (2, 0), (1, 0), (0, 0)]
    kernels = []
    for n in combinations:
        tmp = np.copy(template)
        for m in n:
            tmp[match[m][0], match[m][1]] = 1
        kernels.append(tmp)
    return kernels


def give_intersection_kernels():
    """
    Generates all the intersection kernels in a 9x9 matrix.
    INPUT:
        None
    OUTPUT:
        kernels =      (List) list of 9 x 9 kernels/masks. each element is a mask.
    """
    input_list = np.arange(8)
    taken_n = [4, 3]
    kernels = []
    for taken in taken_n:
        comb = generate_nonadjacent_combination(input_list, taken)
        tmp_ker = populate_intersection_kernel(comb)
        kernels.extend(tmp_ker)
    return kernels


# Find the curve intersections
def find_line_intersection(input_image, show=0):
    """
    Applies morphologyEx with parameter HitsMiss to look for all the curve 
    intersection kernels generated with give_intersection_kernels() function.
    INPUT:
        input_image =  (np.array dtype=np.uint8) binarized m x n image matrix
    OUTPUT:
        output_image = (np.array dtype=np.uint8) image where the nonzero pixels 
                        are the line intersection.
    """
    kernel = np.array(give_intersection_kernels())
    output_image = np.zeros(input_image.shape)
    for i in np.arange(len(kernel)):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
        output_image = output_image + out
    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1), (input_image.shape[0], input_image.shape[1], 3)) * 255
        show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
        show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
        plt.imshow(show_image)
    return output_image, np.array(np.where(output_image == 1)).T


#  finding corners
def find_endoflines(input_image, show=0):
    """
    """
    kernel_0 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, 1, -1]), dtype="int")

    kernel_1 = np.array(([-1, -1, -1], [-1, 1, -1], [1, -1, -1]), dtype="int")

    kernel_2 = np.array(([-1, -1, -1], [1, 1, -1], [-1, -1, -1]), dtype="int")

    kernel_3 = np.array(([1, -1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")

    kernel_4 = np.array(([-1, 1, -1], [-1, 1, -1], [-1, -1, -1]), dtype="int")

    kernel_5 = np.array(([-1, -1, 1], [-1, 1, -1], [-1, -1, -1]), dtype="int")

    kernel_6 = np.array(([-1, -1, -1], [-1, 1, 1], [-1, -1, -1]), dtype="int")

    kernel_7 = np.array(([-1, -1, -1], [-1, 1, -1], [-1, -1, 1]), dtype="int")

    kernel = np.array((kernel_0, kernel_1, kernel_2, kernel_3, kernel_4, kernel_5, kernel_6, kernel_7))
    output_image = np.zeros(input_image.shape)
    for i in np.arange(8):
        out = cv2.morphologyEx(input_image, cv2.MORPH_HITMISS, kernel[i, :, :])
        output_image = output_image + out

    if show == 1:
        show_image = np.reshape(np.repeat(input_image, 3, axis=1), (input_image.shape[0], input_image.shape[1], 3)) * 255
        show_image[:, :, 1] = show_image[:, :, 1] - output_image * 255
        show_image[:, :, 2] = show_image[:, :, 2] - output_image * 255
        plt.imshow(show_image)

    return output_image, np.array(np.where(output_image == 1)).T


# 0- Find end of lines
input_image = skeletons[which].astype(np.uint8)  # must be blaack and white thin network image
eol_img, _eol = find_endoflines(input_image, 0)

# 1- Find curve Intersections
lint_img, _lint = find_line_intersection(input_image, 0)

# 2- Put together all the nodes
nodes = eol_img + lint_img 
plt.imshow(nodes, cmap=plt.cm.gray)

# %%
plt.imshow(skeletons[which] + highlight_edges(input_image, find_line_intersection, scale=10), cmap=plt.cm.gray)

# %%
# cycles: https://stackoverflow.com/questions/15914684/how-can-i-find-cycles-in-a-skeleton-image-with-python-libraries
# symmetry
# average distance to boundary