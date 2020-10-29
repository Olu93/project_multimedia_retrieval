import glob
import sys

import numpy as np
import pandas as pd
import pyqtgraph as pg
import pyvista as pv
from PIL import Image
from PyQt5 import Qt as Qt
from PyQt5 import QtWidgets, QtGui
from PyQt5.QtCore import Qt as QtCore, QUrl
from PyQt5.QtWidgets import QPushButton, QFileDialog, QDesktopWidget, QSlider, QListWidget
from pyvistaqt import QtInteractor
from scipy.spatial.distance import cosine, euclidean, cityblock, sqeuclidean
from scipy.stats import wasserstein_distance

import reader
from feature_extractor import FeatureExtractor
from helper.config import FEATURE_DATA_FILE, DATA_PATH_NORMED, DATA_PATH_PSB
from helper.viz import TableWidget, TsneVisualiser
from normalizer import Normalizer
from query_matcher import QueryMatcher
from reader import DataSet


class SimilarMeshWindow(Qt.QWidget):
    def __init__(self, mesh, features):
        super().__init__()
        self.setWindowTitle('Similar Mesh Window')
        self.mesh = mesh
        self.mesh_features = features
        self.QTIplotter = QtInteractor()
        self.vlayout = Qt.QVBoxLayout()
        self.setLayout(self.vlayout)

        # Create and add widgets to layout
        self.hist_labels = list({**FeatureExtractor.get_pipeline_functions()[1]}.values())
        labels = [f.replace("_", " ").title() for f in list(self.mesh_features.keys())]

        features_df = pd.DataFrame({'key': list(labels), 'value': list(
            [list(f) if isinstance(f, np.ndarray) else f for f in self.mesh_features.values()])}).drop(0)

        # Create Table widget
        self.tableWidget = TableWidget(features_df, self, 5)
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
        screen_topleft = QDesktopWidget().availableGeometry().topLeft()
        screen_height = QDesktopWidget().availableGeometry().height()
        width = (QDesktopWidget().availableGeometry().width() * 0.4)
        self.move((QDesktopWidget().availableGeometry().width() * 0.4) + ((QDesktopWidget().availableGeometry().width() * 0.2)), 0)
        self.resize(width, screen_height - 50)

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
        self.smw_list = []
        self.query_matcher = QueryMatcher(FEATURE_DATA_FILE)
        self.query_mesh_features = feature_dict
        self.layout = Qt.QVBoxLayout()
        self.setLayout(self.layout)
        self.setWindowTitle('Similar Meshes Widget')

        self.scalarDistancesDict = {
            "Cosine": cosine,
            "Manhattan": cityblock,
            "K-Nearest Neighbors": QueryMatcher.perform_knn,
            "Squared Euclidian": sqeuclidean,
            "Euclidean": euclidean
        }

        self.histDistancesDict = {
            "EMD": wasserstein_distance,
            "Cosine": cosine,
            "Manhattan": cityblock,
            "K-Nearest Neighbors": QueryMatcher.perform_knn,
            "Squared Euclidian": sqeuclidean,
            "Euclidean": euclidean
        }

        self.scalarDistanceMethodList = Qt.QComboBox()
        self.scalarDistanceMethodList.addItems(self.scalarDistancesDict.keys())

        self.histDistanceMethodList = Qt.QComboBox()
        self.histDistanceMethodList.addItems(self.histDistancesDict.keys())

        self.sliderK = QSlider(QtCore.Horizontal)
        self.sliderK.setRange(5, 20)
        self.sliderK.valueChanged.connect(self.update_K_label)
        self.Klabel = Qt.QLabel("K: 5", self)

        self.scalarSliderWeights = QSlider(QtCore.Horizontal)
        self.scalarSliderWeights.setRange(0, 10)
        self.scalarSliderWeights.setValue(10)
        self.scalarSliderWeights.valueChanged.connect(self.update_scalar_label)
        self.scalarLabelWeights = Qt.QLabel("Scalars weigh: 1.0", self)

        self.histSliderWeights = QSlider(QtCore.Horizontal)
        self.histSliderWeights.setRange(0, 10)
        self.histSliderWeights.setValue(10)
        self.histSliderWeights.valueChanged.connect(self.update_hist_label)
        self.histLabelWeights = Qt.QLabel("Histogram weight: 1.0", self)

        self.list = QListWidget()
        self.list.setViewMode(Qt.QListView.ListMode)
        self.list.setIconSize(Qt.QSize(150, 150))

        self.matchButton = QPushButton('Match with Database', self)
        self.matchButton.clicked.connect(self.update_similar_meshes_list)

        self.viewTSNEButton = QPushButton('Plot tSNE', self)
        self.viewTSNEButton.clicked.connect(self.plot_tsne)
        self.viewTSNEButton.setToolTip("tSNE computation can take minutes.")
        self.viewTSNEButton.hide()

        self.plotButton = QPushButton('Plot selected mesh', self)
        self.plotButton.clicked.connect(self.plot_selected_mesh)
        self.plotButton.setEnabled(False)
        self.list.currentItemChanged.connect(lambda: self.plotButton.setEnabled(True))

        self.layout.addWidget(Qt.QLabel("Scalar Distance Function", self))
        self.layout.addWidget(self.scalarDistanceMethodList)
        self.layout.addWidget(Qt.QLabel("Histogram Distance Function", self))
        self.layout.addWidget(self.histDistanceMethodList)
        self.layout.addWidget(self.Klabel)
        self.layout.addWidget(self.sliderK)
        self.layout.addWidget(self.scalarLabelWeights)
        self.layout.addWidget(self.scalarSliderWeights)
        self.layout.addWidget(self.histLabelWeights)
        self.layout.addWidget(self.histSliderWeights)
        self.layout.addWidget(self.matchButton)
        self.layout.addWidget(self.plotButton)
        self.layout.addWidget(self.list)

        # Position MainWindow
        screen_height = QDesktopWidget().availableGeometry().height()
        width = (QDesktopWidget().availableGeometry().width() * 0.2)
        self.move((QDesktopWidget().availableGeometry().width() * 0.4), 0)
        self.resize(width, screen_height - 50)

    def closeEvent(self, event):
        self.smw.deleteLater()

    def update_similar_meshes_list(self):
        scalarDistFunction = self.scalarDistancesDict[self.scalarDistanceMethodList.currentText()]
        histDistFunction = self.histDistancesDict[self.histDistanceMethodList.currentText()]
        if (scalarDistFunction or histDistFunction) == QueryMatcher.perform_knn:
            self.scalarDistanceMethodList.setCurrentText("K-Nearest Neighbors")
            self.histDistanceMethodList.setCurrentText("K-Nearest Neighbors")

        #                       OLDER VERSION
        # scalarDistanceFunctionText = self.scalarDistanceMethodList.currentText()
        # scalarDistFunction = self.scalarDistancesDict[scalarDistanceFunctionText]
        #
        # histDistanceFunctionText = self.histDistanceMethodList.currentText()
        # histDistFunction = self.histDistancesDict[histDistanceFunctionText]
        #
        # weights = [self.scalarSliderWeights.value()] + [self.histSliderWeights.value()] * 5
        #
        # features_flattened = QueryMatcher.flatten_feature_dict(self.query_mesh_features)
        # features_df = pd.DataFrame(features_flattened, index=[0])
        # indices, cosine_values = self.query_matcher.compare_features_with_database(features_df,
        #                                                                            weights=weights,
        #                                                                            k=self.sliderK.value(),
        #                                                                            scalar_dist_func=scalarDistFunction,
        #                                                                            hist_dist_func=histDistFunction)

        n_distributionals = len(FeatureExtractor.get_pipeline_functions()[1])

        weights = ([self.scalarSliderWeights.value() / 10]) + \
                  ([self.histSliderWeights.value() / 10] * n_distributionals)

        function_pipeline = [scalarDistFunction] + ([histDistFunction] * n_distributionals)

        indices, distance_values = self.query_matcher.match_with_db(self.query_mesh_features,
                                                                    k=self.sliderK.value(),
                                                                    distance_functions=function_pipeline,
                                                                    weights=weights)

        print(f"Distance values and indices are {list(zip(indices, distance_values))}")

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

        self.viewTSNEButton.show()
        self.layout.addWidget(self.viewTSNEButton)

    def plot_selected_mesh(self):
        mesh_name = self.list.selectedItems()[0].text()
        path_to_mesh = glob.glob(DATA_PATH_NORMED + "\\**\\" + mesh_name + ".*", recursive=True)
        data = DataSet._load_ply(path_to_mesh[0])
        mesh = pv.PolyData(data["vertices"], data["faces"])
        mesh_features = [d for d in self.query_matcher.features_raw if d["name"] == mesh_name][0]
        if len(self.smw_list) != 0:
            self.smw_list[0].deleteLater()
            self.smw_list.remove(self.smw_list[0])
        self.smw = SimilarMeshWindow(mesh, mesh_features)
        self.smw_list.append(self.smw)
        self.smw.show()

    def update_K_label(self, value):
        self.Klabel.setText("KNN: " + str(value))

    def update_scalar_label(self, value):
        self.scalarLabelWeights.setText("Scalar weight: " + str(value / 10))

    def update_hist_label(self, value):
        self.histLabelWeights.setText("Histogram weight: " + str(value / 10))

    def plot_tsne(self):
        tsne_plotter = TsneVisualiser(self.query_matcher.features_raw,
                                      self.query_matcher.features_df,
                                      "tsne.png")
        if not tsne_plotter.file_exist():
            tsne_plotter.plot()
        img = Image.open(open("tsne.png", 'rb'))
        img.show()


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
        self.QTIplotter = None
        self.vlayout = Qt.QVBoxLayout()
        self.frame.setLayout(self.vlayout)
        self.setCentralWidget(self.frame)
        self.hist_dict = {}
        self.setAcceptDrops(True)
        # Create main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # Create load button and init action
        self.load_button = QPushButton("Load or drop mesh to query")
        self.load_button.clicked.connect(lambda: self.load_and_prep_query_mesh(self.open_file_name_dialog()))
        self.load_button.setFont(QtGui.QFont("arial", 30))
        # Create Plots widget
        self.graphWidget = pg.PlotWidget()
        self.graphWidget.setBackground('w')

        # Create and add widgets to layout
        self.hist_labels = list({**FeatureExtractor.get_pipeline_functions()[1]}.values())
        self.tableWidget = TableWidget(pd.DataFrame(), self, len(self.hist_labels))
        self.tableWidget.hide()
        self.vlayout.addWidget(self.load_button)

        # Position MainWindow
        screen_topleft = QDesktopWidget().availableGeometry().topLeft()
        screen_height = QDesktopWidget().availableGeometry().height()
        width = (QDesktopWidget().availableGeometry().width() * 0.4)
        self.move(screen_topleft)
        self.resize(width, screen_height - 50)

        if show:
            self.show()

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.accept()
        else:
            e.ignore()

    def dropEvent(self, e):
        if len(e.mimeData().urls()) == 1:
            file = QUrl.toLocalFile(e.mimeData().urls()[0])
            self.load_and_prep_query_mesh(DataSet._read(file))
        else:
            error_dialog = QtWidgets.QErrorMessage(parent=self)
            error_dialog.showMessage("Please drag only one mesh at the time.")

    @staticmethod
    def check_file(fileName):
        if fileName[-4:] not in self.supported_file_types:
            error_dialog = QtWidgets.QErrorMessage(parent=self)
            error_dialog.showMessage(("Selected file not supported."
                                      f"\nPlease select mesh files of type: {self.supported_file_types}"))
            return False

    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, caption="Choose shape to view.",
                                                  filter="All Files (*);; Model Files (.obj, .off, .ply, .stl)")
        if not (fileName or self.check_file(fileName)):
            return False

        mesh = DataSet._read(fileName)
        return mesh

    def load_and_prep_query_mesh(self, data):
        if not data: return
        self.load_button.setFont(QtGui.QFont("arial", 10))

        # Normalize query mesh
        normed_data = self.normalizer.mono_run_pipeline(data)
        normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"],
                                  normed_data["history"][-1]["data"]["faces"])
        normed_data['poly_data'] = normed_mesh

        # Extract features
        features_dict = FeatureExtractor.mono_run_pipeline(normed_data)
        feature_formatted_keys = [form_key.replace("_", " ").title() for form_key in features_dict.keys()]
        features_df = pd.DataFrame({'key': list(feature_formatted_keys), 'value': list(
            [list(f) if isinstance(f, np.ndarray) else f for f in features_dict.values()])})

        # Update plotter & feature table
        # since unfortunately Qtinteractor which plots the mesh cannot be updated (remove and add new mesh)
        # it needs to be removed and newly generated each time a mesh gets loaded
        self.tableWidget.deleteLater()
        self.vlayout.removeWidget(self.QTIplotter)
        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.QTIplotter)
        self.tableWidget = TableWidget(features_df, self, len(self.hist_labels))
        self.tableWidget.show()
        self.tableWidget.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.vlayout.addWidget(self.tableWidget)
        self.QTIplotter.add_mesh(normed_mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)
        self.vlayout.addWidget(self.graphWidget)

        # Compare shapes
        if self.smlw:
            self.smlw.deleteLater()
            if len(self.smlw.smw_list) != 0:  self.smlw.smw_list[0].deleteLater()
        self.smlw = SimilarMeshesListWindow(features_dict)

        self.hist_dict = features_df.set_index("key").tail().to_dict()
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
