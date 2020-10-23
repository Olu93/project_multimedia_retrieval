# %%
from feature_extractor import FeatureExtractor
from helper.config import DATA_PATH_NORMED_SUBSET
from helper.viz import visualize_histograms

# %%
FE = FeatureExtractor(DATA_PATH_NORMED_SUBSET)
mesh_names = "Ant Human Guitar1 Guitar2".split()
_, hist_functions = FeatureExtractor.get_pipeline_functions()
# %%

# %%
fig = visualize_histograms(FE, hist_functions, item_ids=list(range(4)), names=mesh_names)
fig.savefig('trash/features.png')
# fig.show()
# fig = visualize_hifunctfunctionsroot_volume_four_rand_verts", "Cube root of randomly sampled tetrahedrons",
#                    function_namesange(4)), plot_names)
# fig.savefig('trash/cube_root_volume_four_rand_verts.png')
# fig.show()
# fig = visualize_histograms(FE, "dist_two_rand_verts", "Distance of randomly sampled vertext pairs", list(range(4)),
#                           plot_names)
# fig.savefig('trash/dist_two_rand_verts.png')
# fig.show()
# fig = visualiznum_rows(FE, "dist_bar_vert", "Distance between randomly sampled vertices and barycenter",
#                           linum_rows, plot_names)
# fig.savefig('trash/dist_bar_vert.png')
# fig.show()
# fig = visualize_histograms(FE, "dist_sqrt_area_rand_triangle", "Square root of randomly sampled triangles",
#                           list(range(4)), plot_names)
# fig.savefig('trash/dist_sqrt_area_rand_triangle.png')
# fig.show()
# %%
