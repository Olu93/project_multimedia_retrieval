import glob
import sys

import pandas as pd
import numpy as np
import pyvista as pv
from PyQt5 import Qt as Qt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt as QtCore
from PyQt5.QtWidgets import QPushButton, QFileDialog, QDesktopWidget, QSlider, QListWidget
from pyvistaqt import QtInteractor

import reader
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE, DATA_PATH_NORMED, DATA_PATH_PSB
from helper.viz import TableWidget
from normalizer import Normalizer
from query_matcher import QueryMatcher
from reader import DataSet

df = pd.DataFrame({'x': ['Query mesh description']})


class SimilarMeshWindow(Qt.QWidget):
    def __init__(self, mesh, feature_df):
        super().__init__()
        self.mesh = mesh
        self.mesh_features = feature_df
        self.QTIplotter = QtInteractor()
        self.vlayout = Qt.QVBoxLayout()
        self.setLayout(self.vlayout)
        # self.setCentralWidget(self.frame)

        # Create and add widgets to layout
        # features_df = pd.DataFrame({'key': self.mesh_features.transpose().index,
        #                             'value': list([x[0] for x in self.mesh_features.transpose().values])})

        features_df = pd.DataFrame({'key': list(self.mesh_features.keys()),
                                    'value': list(self.mesh_features.values())}).drop(0)
        self.tableWidget = TableWidget(features_df, self)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vlayout.addWidget(self.QTIplotter.interactor)
        self.vlayout.addWidget(self.tableWidget)

        # Position SimilarMeshWindow
        screen_topRight = QDesktopWidget().availableGeometry().topRight()
        screen_height = QDesktopWidget().availableGeometry().height()
        # self.move(screen_topRight)
        self.resize(self.width(), screen_height - 50)

        # Set widgets
        self.QTIplotter.add_mesh(self.mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)


class SimilarMeshesListWindow(Qt.QWidget):
    def __init__(self, feature_dict):
        super().__init__()
        self.query_matcher = QueryMatcher(FEATURE_DATA_FILE)
        self.query_mesh_features = feature_dict
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self.label = Qt.QLabel("Another Window")

        self.distanceMethodList = Qt.QComboBox()
        self.distanceMethodList.addItems(["Euclidean", "Cosine", "EMD"])

        self.slider1 = QSlider(QtCore.Horizontal)
        self.slider1.setGeometry(30, 40, 200, 30)

        self.sliderKNN = QSlider(QtCore.Horizontal)
        self.sliderKNN.setRange(5, 20)
        self.sliderKNN.valueChanged.connect(self.update_KNN_label)
        self.KNNlabel = Qt.QLabel("KNN: 5", self)

        self.list = QListWidget()
        self.list.setViewMode(Qt.QListView.ListMode)
        self.list.setIconSize(Qt.QSize(150, 150))

        self.matchButton = QPushButton('Match with Database', self)
        self.matchButton.clicked.connect(self.update_similar_meshes_list)

        self.plotButton = QPushButton('Plot selected mesh', self)
        self.plotButton.clicked.connect(self.plot_selected_mesh)
        self.plotButton.setEnabled(False)
        self.list.currentItemChanged.connect(lambda: self.plotButton.setEnabled(True))

        layout.addWidget(self.distanceMethodList)
        layout.addWidget(Qt.QLabel("Distance Slider1"))
        layout.addWidget(self.slider1)
        layout.addWidget(self.KNNlabel)
        layout.addWidget(self.sliderKNN)
        layout.addWidget(self.matchButton)
        layout.addWidget(self.plotButton)
        layout.addWidget(self.list)

    def update_similar_meshes_list(self):
        features_flattened = QueryMatcher.flatten_feature_dict(self.query_mesh_features)
        features_df = pd.DataFrame(features_flattened, index=[0])
        indices, cosine_values = self.query_matcher.compare_features_with_database(features_df,
                                                                                   k=self.sliderKNN.value())
        for ind in indices[0]:
            item = Qt.QListWidgetItem()
            icon = Qt.QIcon()
            filename = 'm' + str(ind) + "_thumb.jpg"
            path_to_thumb = glob.glob(DATA_PATH_PSB + "\\**\\" + filename, recursive=True)
            icon.addPixmap(Qt.QPixmap(path_to_thumb[0]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item.setIcon(icon)
            item.setText("m" + str(ind))
            self.list.addItem(item)

    def plot_selected_mesh(self):
        mesh_name = self.list.selectedItems()[0].text()
        # mesh_name = "m0"
        path_to_mesh = glob.glob(DATA_PATH_NORMED + "\\**\\" + mesh_name + ".*", recursive=True)
        data = DataSet._load_ply(path_to_mesh[0])
        mesh = pv.PolyData(data["vertices"], data["faces"])
        # mesh_features = self.query_matcher.features_raw[self.query_matcher.features_df.index == mesh_name]
        mesh_features = [d for d in self.query_matcher.features_raw if d["name"] == mesh_name][0]
        self.smw = SimilarMeshWindow(mesh, mesh_features)
        self.smw.show()

    def update_KNN_label(self, value):
        self.KNNlabel.setText("KNN: " + str(value))


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.ds = reader.DataSet("")
        self.meshes = []
        self.normalizer = Normalizer()

        self.frame = Qt.QFrame()
        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout = Qt.QVBoxLayout()
        self.frame.setLayout(self.vlayout)
        self.setCentralWidget(self.frame)

        # Create main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Create load button and init action
        load_button = QPushButton("Load Mesh to query")
        load_button.clicked.connect(lambda: self.load_and_prep_query_mesh(self.open_file_name_dialog()))

        # Create and add widgets to layout
        self.tableWidget = TableWidget(df, self)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vlayout.addWidget(load_button)
        self.vlayout.addWidget(self.QTIplotter.interactor)
        self.vlayout.addWidget(self.tableWidget)

        # Position MainWindow
        screen_topleft = QDesktopWidget().availableGeometry().topLeft()
        screen_height = QDesktopWidget().availableGeometry().height()
        self.move(screen_topleft)
        self.resize(self.width(), screen_height - 50)

        if show:
            self.show()

    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self,
                                                  caption="Choose shape to view.",
                                                  filter="All Files (*);; Model Files (.obj, .off, .ply, .stl)")
        if fileName:
            mesh = DataSet._read(fileName)
            return mesh

    def load_and_prep_query_mesh(self, data):
        # Normalize query mesh
        normed_data = self.normalizer.mono_run_pipeline(data)
        if not normed_data: return
        normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"],
                                  normed_data["history"][-1]["data"]["faces"])
        normed_data['poly_data'] = normed_mesh

        # Extract features
        features_dict = FeatureExtractor.mono_run_pipeline(normed_data)
        feature_df = pd.DataFrame({'key': list(features_dict.keys()),
                                   'value': list([list(f) if isinstance(f, np.ndarray)
                                                  else f for f in features_dict.values()])})

        # Update plotter & feature table
        # since unfortunately Qtinteractor which plots the mesh cannot be updated (remove and add new mesh)
        # it needs to be removed and newly generated each time a mesh gets loaded
        self.vlayout.removeWidget(self.tableWidget)
        self.vlayout.removeWidget(self.QTIplotter)
        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.QTIplotter)
        self.tableWidget = TableWidget(feature_df, self)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vlayout.addWidget(self.tableWidget)
        self.QTIplotter.add_mesh(normed_mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)

        # Compare shapes
        self.smlw = SimilarMeshesListWindow(features_dict)
        self.smlw.move(self.geometry().topRight())
        self.smlw.show()


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
