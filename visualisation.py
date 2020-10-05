# %%
from feature_extractor import FeatureExtractor
from helper.viz import visualize_histogram
#%%
FE = FeatureExtractor()
plot_names = "Ant Human Guitar1 Guitar2".split()
# %%
visualize_histogram(FE, "cube_root_volume_four_rand_verts", list(range(4)), plot_names)
visualize_histogram(FE, "angle_three_rand_verts", list(range(4)), plot_names)
visualize_histogram(FE, "dist_two_rand_verts", list(range(4)), plot_names)
visualize_histogram(FE, "dist_bar_vert", list(range(4)), plot_names)
# %%
