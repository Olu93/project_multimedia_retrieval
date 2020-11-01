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
import vtk

import skimage
# %%
mesh = pv.read("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply")
# cow.plot(show_bounds=True, show_grid=True)

# %%
normal = (0, -1, 0)

# %%
projected = mesh.project_points_to_plane((0, 0, 0), normal=normal)
projection_normal = pv.Arrow(direction=normal)


def get_grey_scale(img):
    img_copy = np.ones_like(img)
    img_copy[np.isnan(img)] = 0
    return img_copy


plt.figure()
p = pv.Plotter()
p.add_mesh(projected)
p.camera_position = [(0, -2, 0), (0, 0, 0), (0, 0, 1.0)]
img1 = get_grey_scale(p.get_image_depth())
p.camera_position = [(-2, -2, 0), (0, 0, 0), (0, 0, 1.0)]
img2 = get_grey_scale(p.get_image_depth())

ax = plt.subplot(121)
ax.imshow(img1, cmap=plt.cm.gray)
ax = plt.subplot(122)
ax.imshow(img2, cmap=plt.cm.gray)

plt.show()

# %%
import pyvista as pv
from pyvista import examples

#######
# perfrom the automatic picking
selector = vtk.vtkOpenGLHardwareSelector()
selector.SetFieldAssociation(vtk.vtkDataObject.FIELD_ASSOCIATION_CELLS)
selector.SetRenderer(p.renderer)
selector.SetArea(0, 0, p.window_size[0], p.window_size[1])
selection = selector.Select()

extractor = vtk.vtkExtractSelection()
extractor.SetInputData(0, mesh)
extractor.SetInputData(1, selection)
extractor.Update()

visible = pv.wrap(extractor.GetOutput())

p.add_mesh(visible, color="pink", style="wireframe")
p.show()
# %%
p.get_pick_position
# %%
