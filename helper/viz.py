import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIcon
from PyQt5.QtWidgets import QTableWidget, \
    QTableWidgetItem, QItemDelegate, QLineEdit, QPushButton

from reader import DataSet


def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histograms(extractor, functions, item_ids=[0, 1], names=None, plot_titles=None):
    meshes = np.array(extractor.full_data)[item_ids]
    names = names if names else [data["meta_data"]["label"] for data in meshes]
    plot_titles = plot_titles if plot_titles else list(functions.values())
    result_sets = [[(data, list(func(data).values())[0]) for data in meshes] for func in functions.keys()]
    num_items = len(item_ids)
    num_rows = len(result_sets)
    num_bins = extractor.number_bins
    fig = plt.figure(figsize=(10 * num_items, 9 * num_rows))
    # axes = [
    #     (fig.add_subplot(2, num_items, idx + 1), fig.add_subplot(2, num_items, num_items + idx + 1, projection='3d'))
    #     for idx
    #     in range(num_items)
    #     ]
    hist_axes = fig.subplots(num_rows + 1, num_items, sharex=True)
    #
    for idx, (hist_ax, result_set) in enumerate(zip(hist_axes, result_sets)):

        for ax, (data, results) in zip(hist_ax[:4], result_set):
            ax.bar(np.linspace(0, 1, num_bins), results, 1 / num_bins, align='center')

    for idx, (name, ax, mesh) in enumerate(zip(names, hist_axes[-1, :], meshes)):
        ax.remove()
        last_index = (num_rows * num_items) + idx + 1
        ax = fig.add_subplot(num_rows + 1, num_items, last_index, projection='3d')
        plot_mesh(mesh["poly_data"], ax)

    for ax_row, y_title in zip(hist_axes[:, 0], plot_titles):
        ax_row.set_ylabel(y_title, rotation=90, fontsize=30.0)

    for ax_col, x_title in zip(hist_axes[0, :], names):
        ax_col.set_title(x_title, fontsize=30.0)

    fig.tight_layout()
    # plt.show()
    return fig


class FloatDelegate(QItemDelegate):
    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent=parent)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor


class TableWidget(QTableWidget):
    def __init__(self, df, parent=None, num_non_hist=8):
        QTableWidget.__init__(self, parent)
        self.setEditTriggers(self.NoEditTriggers)
        self.df = df
        nRows = len(self.df.index)
        nColumns = len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)
        self.setItemDelegate(FloatDelegate())
        self.setFixedHeight(250)
        self.buttonsInTable = {}
        key = ""
        value = None

        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                if i < num_non_hist:
                    x = f'{self.df.iloc[i, j]}'
                    self.setItem(i, j, QTableWidgetItem(x))
                else:
                    if j == 1:
                        btn = QPushButton(QIcon('histogramicon.png'), 'Plot Values',self)
                        btn.setText('Plot Values')
                        self.setCellWidget(i, j, btn)
                        value = btn
                    else:
                        x = f'{self.df.iloc[i, j]}'
                        self.setItem(i, j, QTableWidgetItem(x))
                        key = x

                    self.buttonsInTable[key] = value

        self.cellChanged.connect(self.on_cell_changed)

    @pyqtSlot(int, int)
    def on_cell_changed(self, row, column):
        text = self.item(row, column).text()
        number = float(text)
        self.df.set_value(row, column, number)

    def get_buttons_in_table(self):
        return self.buttonsInTable
