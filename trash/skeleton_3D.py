# %%
import numpy as np
from pyvista.core.pointset import StructuredGrid
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
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
import PVGeo


def convex_hull_transformation(mesh):
    poly = mesh
    try:
        hull = ConvexHull(mesh.points)
        poly = PolyData(hull.points).delaunay_3d()
    except QhullError as e:
        print(f"Convex hull operation failed: {str(type(e))} - {str(e)}")
        print("Using fallback!")
    return poly


# %%
mesh_file = "C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\trash\\m1.ply"
mesh = pv.read(mesh_file)
mesh.plot(show_bounds=True, show_grid=True, style="wireframe")
# %%
point_cloud = pv.voxelize(mesh, check_surface=False)
point_cloud.plot()
# %%
selection = point_cloud.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
selection.point_arrays['SelectedPoints']
# %%
# point_cloud = PVGeo.points_to_poly_data(mesh.points)
# voxelizer = PVGeo.filters.VoxelizePoints()
# voxelizer.set_deltas(5, 5, 2) # Your block sizes in dx, dy, dz
# voxelizer.set_estimate_grid(False) # This is crucial for this point cloud
# grid = voxelizer.apply(point_cloud)
# grid.plot(notebook=False)
# %%
skeleton3d = skeletonize_3d(point_cloud.points)
plt.imshow(skeleton3d, cmap=plt.cm.gray, interpolation='nearest')
# %%
density = 42
x_min, x_max, y_min, y_max, z_min, z_max = mesh.bounds
x = np.arange(x_min, x_max, density)
y = np.arange(y_min, y_max, density)
z = np.arange(z_min, z_max, density)
x, y, z = np.meshgrid(x, y, z)
# Create unstructured grid from the structured grid
grid = pv.StructuredGrid(x, y, z)
ugrid = pv.UnstructuredGrid(grid)

# get part of the mesh within the mesh's bounding surface.
selection = ugrid.select_enclosed_points(mesh.extract_surface(), tolerance=0.0, check_surface=False)
mask = selection.point_arrays['SelectedPoints'].view(np.bool_)
ugrid.extract_points(mask)
# %%
from pyntcloud import PyntCloud
import pandas as pd
import numpy as np
from skimage.morphology import skeletonize_3d
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = Axes3D(fig)
voxelcloud = PyntCloud(pd.DataFrame(mesh.points, columns=["x", "y", "z"]), )
voxelcloud = PyntCloud.from_file("C:\\Users\\ohund\\workspace\\project_multimedia_retrieval\\ant.off")


def retrieve_voxel_data(cloud, n=128):
    voxelgrid_id = cloud.add_structure("voxelgrid", n_x=n, n_y=n, n_z=n)
    voxelgrid = cloud.structures[voxelgrid_id]
    vol = np.array(voxelgrid.get_feature_vector(mode="binary"), dtype=np.uint16)
    return vol, voxelgrid


vgrid, vgg = retrieve_voxel_data(voxelcloud)
ax.voxels(vgrid)
plt.show()
# %%
import numpy as np
from skimage.morphology import skeletonize_3d
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = Axes3D(fig)

skeleton3d = skeletonize_3d(vgrid)
ax.voxels(skeleton3d)
plt.show()
# %%
