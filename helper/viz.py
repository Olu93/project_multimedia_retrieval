import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QDoubleValidator
from PyQt5.QtWidgets import QTableWidget, \
    QTableWidgetItem, QItemDelegate, QLineEdit

from reader import DataSet


def plot_mesh(mesh, ax):
    points = mesh.points
    X, Y, Z = points[:, 0], points[:, 1], points[:, 2]
    faces = DataSet._get_cells(mesh)
    return ax.plot_trisurf(X, Y, Z=Z, triangles=faces)


def visualize_histogram(extractor, function_name, plot_title="", item_ids=[0, 1], names=None):
    feature_extraction_function = getattr(extractor, function_name)
    names = names if names else [data["meta_data"]["label"] for data in np.array(extractor.full_data)[item_ids]]
    result_sets = [(data, list(feature_extraction_function(data).values())[0]) for data in
                   np.array(extractor.full_data)[item_ids]]
    num_items = len(result_sets)
    num_bins = extractor.number_bins
    fig = plt.figure(figsize=(5 * num_items, 8))
    axes = [
        (fig.add_subplot(2, num_items, idx + 1), fig.add_subplot(2, num_items, num_items + idx + 1, projection='3d'))
        for idx in range(num_items)]
    for (hist_ax, mesh_ax), (data, results), name in zip(axes, result_sets, names):
        hist_ax.set_title(name)
        hist_ax.bar(np.linspace(0, 1, num_bins), results, 1 / num_bins, align='center')
        plot_mesh(data["poly_data"], mesh_ax)
    fig.suptitle(plot_title, fontsize=20)
    fig.tight_layout()
    return fig


class FloatDelegate(QItemDelegate):
    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent=parent)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor


class TableWidget(QTableWidget):
    def __init__(self, df, parent=None):
        QTableWidget.__init__(self, parent)
        self.df = df
        nRows = len(self.df.index)
        nColumns = len(self.df.columns)
        self.setRowCount(nRows)
        self.setColumnCount(nColumns)
        self.setItemDelegate(FloatDelegate())

        for i in range(self.rowCount()):
            for j in range(self.columnCount()):
                x = f'{self.df.iloc[i, j]}'
                self.setItem(i, j, QTableWidgetItem(x))

        self.cellChanged.connect(self.onCellChanged)

    @pyqtSlot(int, int)
    def onCellChanged(self, row, column):
        text = self.item(row, column).text()
        number = float(text)
        self.df.set_value(row, column, number)
