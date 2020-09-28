import numpy as np
import pyacvd
from pyvista import PolyData
from os import path
import os
from reader import ModelNet40Dataset
from normalizer import Normalizer

norm = Normalizer(ModelNet40Dataset())
norm.scale_to_union()
norm.center()
norm.align()
norm.uniform_remeshing()
norm.save_dataset()
print("Done")