# %%
from feature_extractor import FeatureExtractor
from normalizer import Normalizer
from reader import PSBDataset, DataSet
from helper.config import FEATURE_DATA_FILE, DEBUG, DATA_PATH_NORMED_SUBSET, DATA_PATH_NORMED, CLASS_FILE, DATA_PATH_PSB, DATA_PATH_DEBUG
import time
import pyvista as pv
from pprint import pprint

# %%
if __name__ == "__main__":

    print("=" * 10 + "Testing full pipeline for mono pipeline" + "=" * 10)
    descriptor = DataSet._extract_descr("ant.off")
    data_item = DataSet.mono_run_pipeline(descriptor)
    normed_data_item = Normalizer.mono_run_pipeline(data_item)
    features_data_item = FeatureExtractor.mono_run_pipeline(normed_data_item)

    pprint(features_data_item)

    plotter = pv.Plotter(shape=(1, 2))
    plotter.subplot(0, 0)
    plotter.add_text("Unnormalized", font_size=30)
    plotter.add_mesh(pv.PolyData(data_item["data"]["vertices"], data_item["data"]["faces"]), show_edges=True)
    plotter.show_bounds(all_edges=True)
    plotter.subplot(0, 1)
    plotter.add_text("Normalized", font_size=30)
    plotter.add_mesh(pv.PolyData(normed_data_item["data"]["vertices"], normed_data_item["data"]["faces"]), show_edges=True)
    plotter.show_bounds(all_edges=True)
    plotter.show()

    print("======================================= Done! ===========================================")

    print("=" * 10 + "Testing full pipeline for dataset reader" + "=" * 10)
    dataset = PSBDataset(DATA_PATH_DEBUG, class_file_path=CLASS_FILE)
    dataset.run_full_pipeline()
    dataset.compute_shape_statistics()
    dataset.detect_outliers()
    dataset.convert_all_to_polydata()
    dataset.save_statistics("./trash", "stats_test.csv")
    print("======================================= Done! ===========================================")

    print("=" * 10 + "Testing full pipeline for normalizer" + "=" * 10)
    init_dataset = PSBDataset(DATA_PATH_DEBUG, class_file_path=CLASS_FILE)
    norm = Normalizer(init_dataset)
    norm.target_path = DATA_PATH_NORMED_SUBSET
    normed_data = norm.run_full_pipeline()
    print("======================================= Done! ===========================================")

    print("=" * 10 + "Testing full pipeline for feature extractor" + "=" * 10)
    normed_dataset = PSBDataset(search_path=DATA_PATH_NORMED_SUBSET, class_file_path=CLASS_FILE)
    FE = FeatureExtractor(normed_dataset, target_file="./trash/feat_test.jsonl")
    features = FE.run_full_pipeline()
    print("======================================= Done! ===========================================")
