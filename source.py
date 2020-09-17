import sys

import pyvista as pv
import numpy as np
from PyQt5 import Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog
from pyvistaqt import BackgroundPlotter

import reader


class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        self.meshes = []
        self.plotter = BackgroundPlotter()

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        # self.plotter = QtInteractor(self.frame)
        # vlayout.addWidget(self.plotter.interactor)

        self.frame.setLayout(vlayout)
        self.setCentralWidget(self.frame)

        # simple menu to demo functions
        mainMenu = self.menuBar()
        fileMenu = mainMenu.addMenu('File')
        exitButton = Qt.QAction('Exit', self)
        exitButton.setShortcut('Ctrl+Q')
        exitButton.triggered.connect(self.close)
        fileMenu.addAction(exitButton)

        # allow adding a sphere
        meshMenu = mainMenu.addMenu('Mesh')
        self.add_sphere_action = Qt.QAction('Add Sphere', self)
        self.add_sphere_action.triggered.connect(self.add_sphere)
        meshMenu.addAction(self.add_sphere_action)
        self.add_mesh_action = Qt.QAction('Add Mesh', self)
        self.add_mesh_action.triggered.connect(self.add_mesh)
        meshMenu.addAction(self.add_mesh_action)

        load_button = QPushButton("Load File")
        load_button.clicked.connect(lambda: self.add_mesh(self.open_file_name_dialog()))

        remesh_button = QPushButton("Remesh")
        remesh_button.clicked.connect(lambda: self.remesh())

        vlayout.addWidget(load_button)
        vlayout.addWidget(remesh_button)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Cube()
        self.meshes.append(sphere)
        self.plotter.add_mesh(sphere)
        self.plotter.reset_camera()

    def add_mesh(self, mesh):
        """ add a sphere to the pyqt frame """
        self.meshes.append(mesh)
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()

    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Model Files (.obj, .off, .ply, .stl)")
        ds = reader.DataSet("")
        if fileName:
            if str(fileName).split(".")[1] != "off":
                mesh = ds._load_ply(fileName)
            elif str(fileName).split(".")[1] == "off":
                mesh = ds._load_off(fileName)
            else:
                raise Exception("File type not yet supported.")
            pdmesh = pv.PolyData(mesh["vertices"], mesh["faces"])
            # pdmesh = self.rescale(pdmesh)
            return pdmesh

    def remesh(self):
        globe = self.meshes[-1]
        globe.points *= 0.5

    def rescale(self, mesh):
        max_range = np.max(mesh.points, axis=0)
        min_range = np.min(mesh.points, axis=0)
        lengths_range = max_range - min_range
        longest_range = np.max(lengths_range)
        scaled_points = (mesh.points - min_range) / longest_range
        mesh.points = scaled_points
        return mesh


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
