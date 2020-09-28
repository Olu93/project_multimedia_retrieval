import numpy as np
import pyacvd
from pyvista import PolyData
from os import path
import os
from reader import ModelNet40Dataset
from normalizer import Normalizer
from helper.config import DEBUG, DATA_PATH_MN40, DATA_PATH_DEBUG, CLASS_FILE

norm = Normalizer(ModelNet40Dataset(DATA_PATH_MN40))
norm.run_full_pipeline()
print("Done")