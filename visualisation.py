# %%
from feature_extractor import FeatureExtractor
from helper.viz import visualize_histogram
from helper.config import DATA_PATH_NORMED_SUBSET
#%%
FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET)
plot_names = "Ant Human Guitar1 Guitar2".split()
# %%
visualize_histogram(FE, "angle_three_rand_verts", "Angle of randomly sampled vertext triplets", list(range(4)), plot_names)
visualize_histogram(FE, "cube_root_volume_four_rand_verts", "Cube root of randomly sampled tetrahedrons", list(range(4)), plot_names)
visualize_histogram(FE, "dist_two_rand_verts", "Distance of randomly sampled vertext pairs", list(range(4)), plot_names)
visualize_histogram(FE, "dist_bar_vert", "Distance between randomly sampled vertices and barycenter", list(range(4)), plot_names)
visualize_histogram(FE, "dist_sqrt_area_rand_triangle", "Square root of randomly sampled triangles", list(range(4)), plot_names)
# %%
