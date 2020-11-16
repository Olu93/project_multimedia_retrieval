import os
import webbrowser
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QDoubleValidator, QIcon
from PyQt5.QtWidgets import QTableWidget, \
    QTableWidgetItem, QItemDelegate, QLineEdit, QPushButton
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure, output_file, save
from scipy.stats import entropy
from sklearn.manifold import TSNE

from helper.misc import rand_cmap
from reader import DataSet



class FloatDelegate(QItemDelegate):
    def __init__(self, parent=None):
        QItemDelegate.__init__(self, parent=parent)

    def createEditor(self, parent, option, index):
        editor = QLineEdit(parent)
        editor.setValidator(QDoubleValidator())
        return editor


class TableWidget(QTableWidget):
    def __init__(self, feature_dict, parent, mapping):

        QTableWidget.__init__(self, parent)
        self.setEditTriggers(self.NoEditTriggers)
        self.feature_dict = feature_dict
        self.setFixedHeight(250)
        self.setRowCount(len(mapping))
        self.setItemDelegate(FloatDelegate())
        self.setColumnCount(2)
        self.buttonsInTable = {}
        for idx, (key, values) in enumerate(feature_dict.items()):
            if "hist_" in mapping.get(key, key):
                self.setItem(idx, 0, QTableWidgetItem(key))
                btn = QPushButton(QIcon('figs\histogramicon.png'), 'Plot Values', self)
                btn.setText('Plot Values')

                self.setCellWidget(idx, 1, btn)
                self.buttonsInTable[key] = btn
            if "skeleton_" in mapping.get(key, key):
                self.setItem(idx, 0, QTableWidgetItem(key))
                val = f'X: {round(values[0], 2)}, ' \
                      f'Y: {round(values[1], 2)}, ' \
                      f'Z: {round(values[2], 2)}'
                self.setItem(idx, 1, QTableWidgetItem(val))
            if "scalar_" in mapping.get(key, key):
                self.setItem(idx, 0, QTableWidgetItem(key))
                self.setItem(idx, 1, QTableWidgetItem(str(values)))

        self.cellChanged.connect(self.on_cell_changed)

    @pyqtSlot(int, int)
    def on_cell_changed(self, row, column):
        text = self.item(row, column).text()
        number = float(text)
        self.feature_dict.set_value(row, column, number)

    def get_buttons_in_table(self):
        return self.buttonsInTable


class TsneVisualiser:
    def __init__(self, labels, values, filename, recompute=True):
        self.labels = labels
        self.values = values
        self.filename = filename
        self.recompute = recompute

    def plot(self):
        # If the file exists just show and return
        if self.file_exist():
            webbrowser.open('file://' + os.path.realpath(self.filename))
            return

        # Calculate perplexity
        counts = Counter(self.labels).values()
        probabilities = [prob / len(counts) for prob in counts]
        perplexity = 2**(entropy(probabilities))

        # Evaluate TSNE and create dataframe |labels|tsne_x|tsne_y|
        flat_data = [[val for sublist in row for val in sublist] for row in self.values]
        tsne_results = TSNE(perplexity=1).fit_transform(flat_data)
        t_x, t_y = tsne_results[:, 0], tsne_results[:, 1]
        df = pd.DataFrame(np.hstack((t_x.reshape(-1, 1), t_y.reshape(-1, 1))), index=self.labels)
        df.reset_index(level=0, inplace=True)
        df.columns = ["labels", "x", "y"]

        # Plotting, saving and displaying
        unique_classes = sorted(list(set(df['labels'])))
        classification_indexes = [unique_classes.index(x) for x in df['labels']]
        colors = rand_cmap(len(unique_classes), return_hex=True)  # cc.b_glasbey_bw[0:len(unique_classes)]
        draw_colors = [colors[classification_indexes[x]] for x in range(df.shape[0])]
        TOOLS = [
            "pan", "wheel_zoom", "zoom_in", "zoom_out", "box_zoom", "undo", "redo", "reset", "tap", "save", "box_select", "poly_select", "lasso_select",
            HoverTool(tooltips='@labels')
        ]
        source = ColumnDataSource(data={'x': df["x"], 'y': df["y"], 'labels': df["labels"], 'colors': draw_colors})
        p = figure(title='tSNE', x_axis_label='tSNE 1', y_axis_label='tSNE 2', tools=TOOLS, plot_width=900, plot_height=900)
        p.circle(x='x', y='y', source=source, color="colors", alpha=255)
        output_file(self.filename, title="tSNE Hell.")
        save(p)
        webbrowser.open('file://' + os.path.realpath(self.filename))

    def file_exist(self):
        if os.path.isfile(self.filename):
            if self.recompute:
                os.remove(self.filename)
                return False
            return True
        return False

        # # Calculate perplexity
        # counts = Counter(self.labels).values()
        # probabilities = [prob / len(counts) for prob in counts]
        # perplexity = 2 ** (entropy(probabilities))
        #
        # # Evaluate TSNE and create dataframe |labels|tsne_x|tsne_y|
        # flat_data = [[val for sublist in row for val in sublist] for row in self.values]
        # tsne_results = TSNE(perplexity=perplexity).fit_transform(flat_data)
        # t_x, t_y = tsne_results[:, 0], tsne_results[:, 1]
        # df = pd.DataFrame(np.hstack((np.array(self.labels).reshape(-1, 1),
        #                              np.array(self.labels_coarse).reshape(-1, 1),
        #                              t_x.reshape(-1, 1),
        #                              t_y.reshape(-1, 1))))
        # df.columns = ["labels", "labels_coarse", "x", "y"]
        #
        # # Plotting, saving and displaying
        # unique_classes = sorted(list(set(df['labels_coarse'])))
        # classification_indexes = [unique_classes.index(x) for x in df['labels']]
        # colors = rand_cmap(len(unique_classes), return_hex=True)  # cc.b_glasbey_bw[0:len(unique_classes)]
        # draw_colors = [colors[classification_indexes[x]] for x in range(df.shape[0])]
        # TOOLS = ["pan", "wheel_zoom", "zoom_in", "zoom_out", "box_zoom", "undo", "redo", "reset", "tap", "save",
        #          "box_select", "poly_select", "lasso_select", HoverTool(tooltips=['@labels', '@labels_coarse'])]
        # source = ColumnDataSource(
        #     data={'x': df["x"], 'y': df["y"], 'labels': df["labels"], 'labels_coarse': df["labels_coarse"],
        #           'colors': draw_colors})
        # p = figure(title='tSNE', x_axis_label='tSNE 1', y_axis_label='tSNE 2',
        #            tools=TOOLS, plot_width=900, plot_height=900)
        # p.circle(x='x', y='y', source=source, color="colors", alpha=255)
        # output_file(self.filename, title="tSNE reduction.")
        # save(p)
        # webbrowser.open('file://' + os.path.realpath(self.filename))