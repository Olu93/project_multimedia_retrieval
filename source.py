import sys

import pyvista as pv
from PyQt5 import Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog
from pyvistaqt import BackgroundPlotter

import reader
from normalizer import Normalizer
from reader import DataSet


class MainWindow(Qt.QMainWindow):
    def __init__(self, parent=None, show=True):
        Qt.QMainWindow.__init__(self, parent)

        self.meshes = []
        self.plotter = BackgroundPlotter(shape=(2, 2), border_color='white')

        # create the frame
        self.frame = Qt.QFrame()
        vlayout = Qt.QVBoxLayout()
        # add the pyvista interactor object
        # self.plotter = QtInteractor(self.frame)
        # vlayout.addWidget(self.plotter.interactor)
        self.normalizer = Normalizer()
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
        load_button.clicked.connect(lambda: self.add_mesh(self.open_file_name_dialog()["poly_data"]))

        remesh_button = QPushButton("Show data processing")
        remesh_button.clicked.connect(lambda: self.show_processing(self.open_file_name_dialog()))

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
        self.ds = reader.DataSet("")
        if fileName:
            mesh = DataSet._read(fileName)
            return mesh

    def show_processing(self, mesh):
        new_data = self.normalizer.mono_run_pipeline(mesh)
        history = new_data["history"]
        num_of_operations = len(history)
        plt = BackgroundPlotter(shape=(2, num_of_operations // 2))
        elements = history
        plt.show_axes_all()
        for idx in range(num_of_operations):
            # plt.subplot(0 if idx // 3 == 0 else 1, idx % 3)
            plt.subplot(int(idx / 3), idx % 3)
            if elements[idx]["op"] == "Center":
                plt.add_mesh(pv.Cube().extract_all_edges())
            plt.add_mesh(elements[idx]["data"], color='w', show_edges=True)
            plt.reset_camera()
            plt.view_isometric()
            plt.add_text(
                elements[idx]["op"] + "\nVertices: " + str(len(elements[idx]["data"].points)) + "\nFaces: " + str(
                    elements[idx]["data"].n_faces))
            plt.show_grid()


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
