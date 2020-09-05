import sys

from PyQt5 import Qt
from PyQt5.QtWidgets import QPushButton

import numpy as np

import pyvista as pv
from pyvistaqt import QtInteractor

class MainWindow(Qt.QMainWindow):

    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()

        # add the pyvista interactor object
        self.plotter = QtInteractor(self.frame)
        vlayout.addWidget(self.plotter.interactor)

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

        button = QPushButton("Do not press this!")
        vlayout.addWidget(button)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere)
        self.plotter.reset_camera()

    def add_mesh(self):
        """ add a sphere to the pyqt frame """
        mesh = pv.read("m0.off")
        self.plotter.add_mesh(mesh)
        self.plotter.reset_camera()

if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())