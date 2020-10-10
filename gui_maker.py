import sys
import pandas as pd
import pyvista as pv
from PyQt5 import Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog, QDesktopWidget
from pyvistaqt import BackgroundPlotter, QtInteractor
import reader
from helper.viz import TableWidget
from normalizer import Normalizer
from reader import DataSet
from feature_extractor import FeatureExtractor

df = pd.DataFrame({'x': ['Query mesh description']})


class AnotherWindow(Qt.QWidget):
    """
    This "window" is a QWidget. If it has no parent, it
    will appear as a free-floating window as we want.
    """

    def __init__(self):
        super().__init__()
        layout = Qt.QVBoxLayout()
        self.label = Qt.QLabel("Another Window")
        layout.addWidget(self.label)
        self.setLayout(layout)


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

        #Create main menu
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        #Create load button and init action
        load_button = QPushButton("Load Mesh to query")
        load_button.clicked.connect(lambda: self.load_and_prep_query_mesh(self.open_file_name_dialog()))

        #Create and add widgeds to layout
        self.tableWidget = TableWidget(df, self)
        self.vlayout.addWidget(load_button)
        self.vlayout.addWidget(self.QTIplotter.interactor)
        self.vlayout.addWidget(self.tableWidget)

        #Position MainWindow
        screen_topleft = QDesktopWidget().availableGeometry().topLeft()
        screen_height = QDesktopWidget().availableGeometry().height()
        self.move(screen_topleft)
        self.resize(self.width(), screen_height-50)

        if show:
            self.show()



    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "Choose shape to view.", "",
                                                  "All Files (*);; Model Files (.obj, .off, .ply, .stl)")
        if fileName:
            mesh = DataSet._read(fileName)
            return mesh

    def load_and_prep_query_mesh(self, data):
        #Normalize query mesh
        normed_data = self.normalizer.mono_run_pipeline(data)
        normed_mesh = pv.PolyData(normed_data["history"][-1]["data"]["vertices"], normed_data["history"][-1]["data"]["faces"])
        normed_data['poly_data'] = normed_mesh

        #Extract features
        features_dict = FeatureExtractor.mono_run_pipeline(normed_data)
        feature_df = pd.DataFrame({'key': list(features_dict.keys()), 'value': list(features_dict.values())})

        #Update plotter & feature table
        self.vlayout.removeWidget(self.tableWidget)
        self.vlayout.removeWidget(self.QTIplotter)

        self.QTIplotter = QtInteractor(self.frame)
        self.vlayout.addWidget(self.QTIplotter)

        self.tableWidget = TableWidget(feature_df, self)
        self.vlayout.addWidget(self.tableWidget)

        self.QTIplotter.add_mesh(normed_mesh, show_edges=True)
        self.QTIplotter.isometric_view()
        self.QTIplotter.show_bounds(grid='front', location='outer', all_edges=True)

        #Compare shaped
        self.show_similar_meshes_list()

    def show_similar_meshes_list(self):
        w = AnotherWindow()
        w.show()



if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())