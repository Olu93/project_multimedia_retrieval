import sys

import pyvista as pv
from PyQt5 import Qt
from PyQt5.QtWidgets import QPushButton, QFileDialog
from pyvistaqt import QtInteractor


def ply_to_off(file_name):
    with open(file_name) as file:
        vertices = []
        faces = []
        header_ended = False
        for i, line in enumerate(file.readlines()):
            if "vertex " in line:
                num_vertices = int(line.split("vertex")[1])
                continue
            elif "face" in line:
                num_faces = int(line.split("face")[1])
                continue
            elif "end_header" in line:
                end = (i + 1) + num_vertices
                header_ended = True
                continue
            if header_ended:
                if i < end:
                    vertices.append(line)
                else:
                    faces.append(line)
    new_file_name = file_name.split(".")[0] + ".off"
    new_file = open(new_file_name, "w")
    new_file.write("OFF\n")
    new_file.write(str(num_vertices) + " " + str(num_faces) + " " + "0\n")
    new_file.writelines(vertices)
    new_file.writelines(faces)
    return new_file_name


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

        button = QPushButton("Load File")
        button.clicked.connect(lambda: self.add_mesh(self.open_file_name_dialog()))

        vlayout.addWidget(button)

        if show:
            self.show()

    def add_sphere(self):
        """ add a sphere to the pyqt frame """
        sphere = pv.Sphere()
        self.plotter.add_mesh(sphere)
        self.plotter.reset_camera()

    def add_mesh(self, path):
        """ add a sphere to the pyqt frame """
        if path:
            mesh = pv.read(path)
            self.plotter.add_mesh(mesh)
            self.plotter.reset_camera()

    def open_file_name_dialog(self):
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Model Files (.obj, .off, .ply, .stl)")
        if fileName:
            if str(fileName).split(".")[1] != "off":
                return ply_to_off(fileName)
            return fileName


if __name__ == '__main__':
    app = Qt.QApplication(sys.argv)
    window = MainWindow()
    sys.exit(app.exec_())
