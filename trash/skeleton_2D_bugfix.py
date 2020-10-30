# %%
from PIL.Image import Image
import numpy as np
from numpy.core.defchararray import center
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
import cv2
import networkx as nx
import pandas as pd
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")

# mesh.plot()


# %%
def prepare_image(img, proj):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy, proj


def extract_sillhouettes(mesh):
    images = []
    normal = np.zeros((3, 1))
    p = pv.Plotter(
        notebook=False,
        off_screen=True,
    )
    for i in range(3):
        normal[:] = 0
        normal[i] = -1
        projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
        p.add_mesh(projected)
        p.set_position(normal * 2)
        img = p.get_image_depth()
        images.append(prepare_image(img, projected))
    return images


def extract_skeletons(sillh):
    return [binary_closing(skeletonize(img_array, method="zhang")).astype(np.uint8) for img_array in sillh]


def extract_graphs(skeletons):
    graphs = [sknw.build_sknw(ske) for ske in skeletons]
    for G in graphs:
        G.remove_nodes_from(list(nx.isolates(G)))
    return graphs


sillhouettes = extract_sillhouettes(mesh)
plt.imshow(sillhouettes[2][0], 'gray')
# %%
projected = sillhouettes[2][1]
dims = np.array([1024, 724])
img = projected.points
not_null = np.sum(img, axis=0) != 0
points = img[:, not_null]

positive_points = points - points.min(axis=0)
scaled_points = positive_points / positive_points.max(axis=0)
retransformed_points = scaled_points * dims
points_on_canvas = np.floor(retransformed_points).astype(int)
canvas = np.zeros(dims + 1)
canvas[points_on_canvas[:, 0], points_on_canvas[:, 1]] = 1
plt.imshow(canvas.T, cmap="gray")

# %%

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
from skimage import data, color

fig = Figure()
canvas = FigureCanvas(fig)
ax = fig.gca()

ax.scatter(points_on_canvas[:, 0], points_on_canvas[:, 1], c='k')
ax.axis('off')

canvas.draw()
buf = canvas.buffer_rgba()
X = np.asarray(buf)
gray_image = X.mean(axis=2)
normalized = (gray_image - gray_image.min()) / (gray_image.max()-gray_image.min())
pre_prep = 1 - normalized
pre_prep[pre_prep != 0] = 1
plt.imshow(pre_prep, cmap='gray')
# %%
import sys
from descartes import PolygonPatch
import matplotlib.pyplot as plt
import alphashape
alpha_shape = alphashape.alphashape(np.flip(positive_points), 100)
fig, ax = plt.subplots()
contour_of_points = np.array(alpha_shape.exterior.coords)
# ax.scatter(*zip(*positive_points))
ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))
plt.show()

# %%


# # normalize the data and convert to uint8 (grayscale conventions)
# zNorm = (Z - Z.min()) / (Z.max() - Z.min()) * 255
# zNormUint8 = zNorm.astype(np.uint8)

# # plot result
# plt.figure()
# plt.imshow(zNormUint8)
from scipy.interpolate import griddata
contour_of_points_remapped = contour_of_points * 100
df = pd.DataFrame(points_on_canvas, columns=["X", "Y"])
x_range = ((df.X.max() - df.X.min()))
y_range = ((df.Y.max() - df.Y.min()))
grid_x, grid_y = np.mgrid[df.X.min():df.X.max():(x_range * 1j), df.Y.min():df.Y.max():(y_range * 1j)]
points = df[['X', 'Y']].values
values = np.ones(len(df))
grid_z0 = griddata(points, values, (grid_x, grid_y)).astype(np.uint8)
plt.imshow(grid_z0, cmap='gray')
# im.show()

# %%
fig, ax = plt.subplots()
ax.imshow(canvas, cmap=plt.cm.gray)
contours = skimage.measure.find_contours(canvas, .1)
ax.plot(contours[0][:, 1], contours[0][:, 0], linewidth=2)
# plt.scatter(retransformed_points[:,0], retransformed_points[:,1])
# %%
from polylidar import extractPlanesAndPolygons, extractPolygons, Delaunator

kwargs = dict(alpha=0.0, lmax=1.0)

# You want everything!
delaunay, planes, polygons = extractPlanesAndPolygons(positive_points, **kwargs)

# Show me JUST the polygons!
polygons = extractPolygons(positive_points, **kwargs)

# Also if you just want fast 2D delaunay triangulation, no polylidar
delaunay = Delaunator(positive_points)

# %%
important_points = positive_points[polygons[0].shell]

# %%
from 
tri = Delaunay(positive_points)
plt.scatter(positive_points[:,0], positive_points[:,1], tri.simplices)
X, Y = np.meshgrid(positive_points[:,0], positive_points[:,1])
Z = X ** 2 + Y ** 2

# %%
from shapely.geometry import Polygon
polygon = Polygon(positive_points)
polygon

# %%
import numpy
from PIL import Image, ImageDraw

# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?
img = Image.new('L', (500, 500), 0)
drawing = ImageDraw.Draw(img)
drawing.polygon(np.array(alpha_shape.exterior.coords), outline=1, fill=255)
mask = numpy.array(img)
img


# %%
import PIL.ImageDraw as ImageDraw
import PIL.Image as Image

image = Image.new("RGB", (640, 480))

draw = ImageDraw.Draw(image)

# points = ((1,1), (2,1), (2,2), (1,2), (0.5,1.5))
points = ((100, 100), (200, 100), (200, 200), (100, 200), (50, 150))
draw.polygon((points), fill=200)

image.show()
# %%
grid_data = np.mgrid[0:1:100j, 0:1:100j]
