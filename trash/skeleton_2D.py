# %%
import numpy as np
from scipy.spatial import ConvexHull
from pyvista import PolyData
from pyvista import examples
import pyvista as pv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from skimage.util import dtype
from skimage.morphology import skeletonize, binary_dilation, binary_erosion, skeletonize_3d
from skimage.util import invert
import skimage
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")
# cow.plot(show_bounds=True, show_grid=True)

# %%
normal = (0, -1, 0)

# %%
projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
projection_normal = pv.Arrow(direction=normal)
# projected.plot()
# %%
# %%
p = pv.Plotter()
p.show_bounds()
p.show_grid()
p.add_mesh(mesh.outline(), color="k")
p.camera_position = [(2, -2, 0), (0.0, 0, 0), (0, 0, 1.0)]
p.add_mesh(mesh, style="wireframe")
# p.add_mesh(projected)
# p.add_mesh(projection_normal)
p.show()

# %%


def get_grey_scale(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


plt.figure()
p = pv.Plotter()
p.add_mesh(projected)
p.camera_position = [(0, -2, 0), (0, 0, 0), (0, 0, 1.0)]
img1 = get_grey_scale(p.get_image_depth())
p.camera_position = [(-2, 0, 0), (0, 0, 0), (0, 0, 1.0)]
img2 = get_grey_scale(p.get_image_depth())
p.camera_position = [(0, 0, -2), (0, 0, 0), (0, 0, 1.0)]
img3 = get_grey_scale(p.get_image_depth())

ax = plt.subplot(121)
ax.imshow(img, cmap=plt.cm.gray)
ax = plt.subplot(122)
ax.imshow(img2, cmap=plt.cm.gray)

plt.show()

# %%

# %%
skeleton = skeletonize(img_copy)
plt.imshow(skeleton, cmap=plt.cm.gray)
plt.show()

# %%
# import cv
# gray = cv.cvtColor(skeleton,cv2.COLOR_BGR2GRAY)
# ret, thresh = cv2.threshold(gray, 127, 255, 1)
# cv2.imshow(ret)
# %%
from skimage.filters import gaussian
dilated_skeleton = binary_dilation(skeleton)
eroded_skeleton = binary_erosion(dilated_skeleton)
plt.imshow(eroded_skeleton, cmap=plt.cm.gray)
plt.show()
# %%
from skimage.morphology import binary_closing, convex_hull_image, convex_hull_object
closed_skeleton = binary_closing(skeleton)
plt.imshow(closed_skeleton, cmap=plt.cm.gray)
plt.show()

# %%
plt.imshow(convex_hull_object(img_copy), cmap=plt.cm.gray)
plt.show()
# %%
from skimage.morphology import area_closing, remove_small_holes

plt.imshow(remove_small_holes(np.array(img_copy, dtype=np.int64)), cmap=plt.cm.gray)
plt.show()