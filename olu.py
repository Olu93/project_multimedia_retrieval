from feature_extractor import FeatureExtractor
import time
if __name__ == "__main__":
    FE = FeatureExtractor()
    start = time.time()
    FE.diameter2(FE.full_data[0])
    end = time.time() - start
    print(f"Roughly {end} secs.")