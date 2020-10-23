from feature_extractor import FeatureExtractor
import sys

import pandas as pd
import pyvista as pv
from PyQt5 import Qt
from PyQt5.QtWidgets import QFileDialog
from pyvistaqt import BackgroundPlotter

import reader
from helper.viz import TableWidget
from normalizer import Normalizer
from reader import DataSet


# df = pd.DataFrame({'a': ['Mary', 'Jim', 'John'],
#                    'b': [100, 200, 300],
#                    'c': ['a', 'b', 'c']})


# TODO: [x] Fix bug, when "cancel" pressed on file choosing window
# TODO: [] Add Tabular viewer from QT and link it to data table from feature extractor


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.ds = reader.DataSet("")
        self.meshes = []
        self.plotter = BackgroundPlotter(shape=(1, 2), border_color='white', title="MMR Visualization")
        self.setWindowTitle('MMR UI')
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()
        self.normalizer = Normalizer()
        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)
        meshMenu = mainMenu.addMenu('Mesh')

        self.load_mesh = Qt.QAction('Load mesh', self)
        self.load_mesh.triggered.connect(lambda: self.add_mesh(self.open_file_name_dialog()))
        meshMenu.addAction(self.load_mesh)

        self.show_norm_pipeline = Qt.QAction('Show norm pipeline', self)
        self.show_norm_pipeline.triggered.connect(lambda: self.show_processing(self.open_file_name_dialog()))
        meshMenu.addAction(self.show_norm_pipeline)

        self.extract_features = Qt.QAction('Extract features', self)
        self.extract_features.triggered.connect(lambda: print(FeatureExtractor.mono_run_pipeline(self.open_file_name_dialog())))
        meshMenu.addAction(self.extract_features)


        if show:
            self.show()

    def add_mesh(self, mesh):
        if not mesh:
            print(f"Can't render object of type {type(mesh)}")
            return None

        self.meshes.append(mesh["poly_data"])
        self.plotter.add_mesh(mesh["poly_data"])
        df = pd.DataFrame.from_dict(self.fe.mono_run_pipeline(mesh))
        self.tableWidget = TableWidget(df, self)
        self.frame.layout().addWidget(self.tableWidget)
        self.plotter.reset_camera()

    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  caption="Choose shape to view.",
                                                  filter="All Files (*);; Model Files (.obj, .off, .ply, .stl)")
        if fileName:
            mesh = DataSet._read(fileName)
            return mesh
        return None

    def show_processing(self, mesh):
        if not mesh:
            print(f"Can't render mesh of type {type(mesh)}")
            return None

        new_data = self.normalizer.mono_run_pipeline(mesh)
        history = new_data["history"]
        num_of_operations = len(history)
        plt = BackgroundPlotter(shape=(2, num_of_operations // 2))
        elements = history
        plt.show_axes_all()
        for idx in range(num_of_operations):
            plt.subplot(int(idx / 3), idx % 3)
            if elements[idx]["op"] == "Center":
                plt.add_mesh(pv.Cube().extract_all_edges())
            curr_mesh = pv.PolyData(elements[idx]["data"]["vertices"], elements[idx]["data"]["faces"])
            plt.add_mesh(curr_mesh, color='w', show_edges=True)
            plt.reset_camera()
            plt.view_isometric()
            plt.add_text(
                elements[idx]["op"] +
                "\nVertices: " + str(len(curr_mesh.points)) +
                "\nFaces: " + str(curr_mesh.n_faces))
            plt.show_grid()


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
