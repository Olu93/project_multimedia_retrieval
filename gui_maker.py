import glob
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
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
from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance
df = pd.DataFrame({'x': ['Query mesh description']})


class SimilarMeshWindow(Qt.QWidget):
    def __init__(self, mesh, feature_df):
        super().__init__()
        self.setWindowTitle('Similar Mesh Window')
        self.mesh = mesh
        self.mesh_features = feature_df
        self.QTIplotter = QtInteractor()
        self.vlayout = Qt.QVBoxLayout()
        self.setLayout(self.vlayout)

        # Create and add widgets to layout
        # features_df = pd.DataFrame({'key': self.mesh_features.transpose().index,
        #                             'value': list([x[0] for x in self.mesh_features.transpose().values])})

        feature_formatted_keys = [form_key.replace("_", " ").title() for form_key in self.mesh_features.keys()]

        features_df = pd.DataFrame({'key': list(feature_formatted_keys), 'value': list([list(f) if isinstance(f, np.ndarray) else f for f in self.mesh_features.values()])}).drop(0)

        # Create Table widget
        self.tableWidget = TableWidget(features_df, self, 8)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)

        # Create Plots widget
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')

        self.hist_dict = features_df.set_index("key").tail().to_dict()
        self.buttons = self.tableWidget.get_buttons_in_table()

        for key, value in self.buttons.items():
            value.clicked.connect(lambda state, x=key, y=self.hist_dict["value"][key]: self.plot_selected_hist(x, y))

        self.vlayout.addWidget(self.QTIplotter.interactor)
        self.vlayout.addWidget(self.tableWidget)
        self.vlayout.addWidget(self.graphWidget)

        # Position SimilarMeshWindow
        screen_height = QDesktopWidget().availableGeometry().height()
        self.resize(self.width(), screen_height - 50)

        # Set widgets
        self.QTIplotter.add_mesh(self.mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)

    def plot_selected_hist(self, hist_title, hist_data):
        self.graphWidget.clear()
        styles = {"color": "#f00", "font-size": "15px"}
        pen = pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.SolidLine)
        self.graphWidget.setTitle(hist_title, color="b", size="15pt")
        self.graphWidget.setLabel("left", "Values", **styles)
        self.graphWidget.setLabel("bottom", "Bins", **styles)
        self.graphWidget.addLegend()
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setXRange(0, len(hist_data))
        self.graphWidget.setYRange(min(hist_data), max(hist_data))
        self.graphWidget.plot(np.arange(0, len(hist_data)), hist_data, pen=pen)


class SimilarMeshesListWindow(Qt.QWidget):
    def __init__(self, feature_dict):
        super().__init__()
        self.query_matcher = QueryMatcher(FEATURE_DATA_FILE)
        self.query_mesh_features = feature_dict
        layout = Qt.QVBoxLayout()
        self.setLayout(layout)
        self.setWindowTitle('Similar Meshes Widget')

        self.scalarDistancesDict = {  # "Handmade Cosine": QueryMatcher.cosine_similarity_faf,
            "EMD": QueryMatcher.wasserstein_distance,
            "Cosine": QueryMatcher.cosine_distance,
            "Manhattan": QueryMatcher.manhattan_distance,
            "K-Nearest Neighbors": QueryMatcher.perform_knn,
            "Squared Euclidian": QueryMatcher.sqeuclidean_distance,
            "Euclidean": QueryMatcher.euclidean_distance
        }

        self.scalarDistanceMethodList = Qt.QComboBox()
        self.scalarDistanceMethodList.addItems(self.scalarDistancesDict.keys())

        self.sliderKNN = QSlider(QtCore.Horizontal)
        self.sliderKNN.setRange(5, 20)
        self.sliderKNN.valueChanged.connect(self.update_K_label)
        self.KNNlabel = Qt.QLabel("K: 5", self)

        self.list = QListWidget()
        self.list.setViewMode(Qt.QListView.ListMode)
        self.list.setIconSize(Qt.QSize(150, 150))

        self.matchButton = QPushButton('Match with Database', self)
        self.matchButton.clicked.connect(self.update_similar_meshes_list)

        self.plotButton = QPushButton('Plot selected mesh', self)
        self.plotButton.clicked.connect(self.plot_selected_mesh)
        self.plotButton.setEnabled(False)
        self.list.currentItemChanged.connect(lambda: self.plotButton.setEnabled(True))

        layout.addWidget(self.scalarDistanceMethodList)
        layout.addWidget(self.KNNlabel)
        layout.addWidget(self.sliderKNN)
        layout.addWidget(self.matchButton)
        layout.addWidget(self.plotButton)
        layout.addWidget(self.list)

    def update_similar_meshes_list(self):
        # scalarDistanceFunctionText = self.scalarDistanceMethodList.currentText()
        # scalarDistFunction = self.scalarDistancesDict[scalarDistanceFunctionText]
        # features_df = pd.DataFrame(features_flattened, index=[0])
        function_pipeline = [euclidean] + ([wasserstein_distance] * (len(self.query_matcher.features_list_of_list[0]) - 1))
        weights = [.5] + ([.1] * (len(self.query_matcher.features_list_of_list[0]) - 1))
        indices, cosine_values = self.query_matcher.match_with_db(self.query_mesh_features, k=self.sliderKNN.value(), distance_functions=function_pipeline, weights=weights)
        self.list.clear()
        for ind in indices:
            item = Qt.QListWidgetItem()
            icon = Qt.QIcon()
            filename = str(ind) + "_thumb.jpg"
            path_to_thumb = glob.glob(DATA_PATH_PSB + "\\**\\" + filename, recursive=True)
            icon.addPixmap(Qt.QPixmap(path_to_thumb[0]), QtGui.QIcon.Normal, QtGui.QIcon.Off)
            item.setIcon(icon)
            item.setText(str(ind))
            self.list.addItem(item)

    def plot_selected_mesh(self):
        mesh_name = self.list.selectedItems()[0].text()
        path_to_mesh = glob.glob(DATA_PATH_NORMED + "\\**\\" + mesh_name + ".*", recursive=True)
        data = DataSet._load_ply(path_to_mesh[0])
        mesh = pv.PolyData(data["vertices"], data["faces"])
        mesh_features = [d for d in self.query_matcher.features_raw if d["name"] == mesh_name][0]
        self.smw = SimilarMeshWindow(mesh, mesh_features)
        self.smw.show()

    def update_K_label(self, value):
        self.KNNlabel.setText("KNN: " + str(value))


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)
        self.supported_file_types = [".ply", ".off"]
        self.buttons = {}
        self.ds = reader.DataSet("")
        self.meshes = []
        self.normalizer = Normalizer()
        self.smlw = None
        self.setWindowTitle('Source Mesh Window')

        self.frame = Qt.QFrame()
        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout = Qt.QVBoxLayout()
        self.frame.setLayout(self.vlayout)
        self.setCentralWidget(self.frame)
        self.hist_dict = {}

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

        # Create Plots widget
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')

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
        fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose shape to view.", filter="All Files (*);; Model Files (.obj, .off, .ply, .stl)")
        if not fileName:
            return False
        elif fileName[-4:] not in self.supported_file_types:
            error_dialog = QtWidgets.QErrorMessage(parent=self)
            error_dialog.showMessage(("Selected file not supported." f"\nPlease select mesh files of type: {self.supported_file_types}"))
            return False

        mesh = DataSet._read(fileName)
        return mesh

    def load_and_prep_query_mesh(self, data):
        if not data: return
        # Normalize query mesh
        normed_data = self.normalizer.mono_run_pipeline(data)
        normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"], normed_data["history"][-1]["data"]["faces"])
        normed_data['poly_data'] = normed_mesh

        # Extract features
        features_dict = FeatureExtractor.mono_run_pipeline(normed_data)
        feature_formatted_keys = [form_key.replace("_", " ").title() for form_key in features_dict.keys()]
        feature_df = pd.DataFrame({'key': list(feature_formatted_keys), 'value': list([list(f) if isinstance(f, np.ndarray) else f for f in features_dict.values()])})

        # Update plotter & feature table
        # since unfortunately Qtinteractor which plots the mesh cannot be updated (remove and add new mesh)
        # it needs to be removed and newly generated each time a mesh gets loaded
        self.vlayout.removeWidget(self.tableWidget)
        self.vlayout.removeWidget(self.QTIplotter)
        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.QTIplotter)
        self.tableWidget = TableWidget(feature_df.drop(1, axis='rows'), self)
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vlayout.addWidget(self.tableWidget)
        self.QTIplotter.add_mesh(normed_mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)
        self.vlayout.addWidget(self.graphWidget)

        # Compare shapes
        self.smlw = SimilarMeshesListWindow(features_dict)
        self.smlw.move(self.geometry().topRight())

        self.hist_dict = feature_df.set_index("key").tail().to_dict()
        self.buttons = self.tableWidget.get_buttons_in_table()

        for key, value in self.buttons.items():
            value.clicked.connect(lambda state, x=key, y=self.hist_dict["value"][key]: self.plot_selected_hist(x, y))

        self.smlw.show()

    def plot_selected_hist(self, hist_title, hist_data):
        self.graphWidget.clear()
        styles = {"color": "#f00", "font-size": "15px"}
        pen = pg.mkPen(color=(255, 0, 0), width=5, style=QtCore.SolidLine)
        self.graphWidget.setTitle(hist_title, color="b", size="15pt")
        self.graphWidget.setLabel("left", "Values", **styles)
        self.graphWidget.setLabel("bottom", "Bins", **styles)
        self.graphWidget.addLegend()
        self.graphWidget.showGrid(x=True, y=True)
        self.graphWidget.setXRange(1, len(hist_data))
        self.graphWidget.setYRange(min(hist_data), max(hist_data))
        self.graphWidget.plot(np.arange(0, len(hist_data)), hist_data, pen=pen)


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
