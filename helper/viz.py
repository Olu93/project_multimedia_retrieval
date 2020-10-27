import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIcon
from PyQt5.QtWidgets import QTableWidget, \
    QTableWidgetItem, QItemDelegate, QLineEdit, QPushButton
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from helper.misc import rand_cmap
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
    def __init__(self, df, parent, num_hist):
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
                if i < self.rowCount() - num_hist:
                    x = f'{self.df.iloc[i, j]}'
                    self.setItem(i, j, QTableWidgetItem(x))
                else:
                    if j == 1:
                        btn = QPushButton(QIcon('histogramicon.png'), 'Plot Values', self)
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


class TsneVisualiser:
    def __init__(self, raw_data, full_mat, filename):
        self.raw_data = raw_data
        self.full_mat = full_mat.values
        self.filename = filename

    def plot(self):
        labelled_mat = np.hstack(
            (np.array([dic["label"] for dic in self.raw_data]).reshape(-1, 1), self.full_mat))
        df = pd.DataFrame(data=labelled_mat[:, 1:],
                          index=labelled_mat[:, 0])

        lbl_list = list(df.index)
        color_map = rand_cmap(len(lbl_list), first_color_black=False, last_color_black=True)
        lbl_to_idx_map = dict(zip(lbl_list, range(len(lbl_list))))
        labels = [lbl_to_idx_map[i] for i in lbl_list]

        scaler = StandardScaler()
        st_values = scaler.fit_transform(df.values)

        # Playing around with parameters, this seems like a good fit
        tsne_results = TSNE(perplexity=40, learning_rate=500).fit_transform(st_values)
        t_x, t_y = tsne_results[:, 0], tsne_results[:, 1]
        plt.title("tSNE reduction")
        plt.xlabel("tSNE 1")
        plt.ylabel("tSNE 2")
        plt.scatter(t_x, t_y, c=labels, cmap=color_map, vmin=0, vmax=len(lbl_list), label=lbl_list, s=10)
        plt.savefig(self.filename, bbox_inches='tight', dpi=200)

    def file_exist(self):
        if os.path.isfile(self.filename):
            return True
        return False
