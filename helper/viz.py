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
from sklearn.decomposition import PCA

from helper.misc import rand_cmap
from reader import DataSet

VERBOSE = True


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


def prepare_data(query_matcher, top_n=0, reverse=False, strange_scaling=False, is_coarse=True):
    flat_data = query_matcher.features_df_properly_scaled
    all_lbls = [query_matcher.map_to_label(name, is_coarse) for name in list(flat_data.index)]
    final_data = flat_data

    if strange_scaling:
        flat_data = query_matcher.features_df_all_scaled
    if top_n:
        flat_data["label"] = all_lbls
        all_lbls_cnt = Counter(all_lbls)
        # resversed_order_bit = -1 if reverse else 1
        lbl_count_df = pd.DataFrame(all_lbls_cnt.most_common(), columns=("label", "cnt"))
        sum_top_lbls = lbl_count_df[:top_n].cnt.sum()
        ratio = sum_top_lbls / flat_data.shape[0]
        cutoff_point = int(ratio * lbl_count_df.cnt.sum())

        top_lbls = {}
        selected_lbls = None
        for key, cnt in lbl_count_df.sort_values(by="cnt", ascending=reverse).values:

            top_lbls[key] = cnt
            if sum(top_lbls.values()) >= cutoff_point:
                selected_lbls = pd.DataFrame(top_lbls.items(), columns=("label", "cnt"))
                break
        # lbl_count_df.sort_values(by="cnt", ascending=reverse).iloc[:cutoff_point]
        min_lbl_cnt = selected_lbls.cnt.min()

        final_data = pd.concat([flat_data[flat_data.label == key].sample(min_lbl_cnt) for key, _ in selected_lbls.values]).drop("label", axis=1)

    return final_data


def compute_tsne(flat_data, perplexity, n_iter, lr, names_n_labels, is_coarse=False, d=2, num_jobs=None) -> pd.DataFrame:
    tsne_results = TSNE(d, perplexity=perplexity, early_exaggeration=1, n_iter=int(n_iter), learning_rate=lr, n_jobs=num_jobs, verbose=VERBOSE).fit_transform(flat_data)
    all_together = np.array([x + list(y) for x, y in zip(names_n_labels, tsne_results)], dtype=object)
    cols = ["label", "x", "y"]
    if d == 3:
        cols = ["label", "x", "y", "z"]
    if d > 3:
        cols = ["label"] + [f"x{idx}" for idx in range(d)]
    df = pd.DataFrame(all_together[:, 1:], index=all_together[:, 0], columns=cols)
    return df


def compute_tsne_subset(query_matcher, perplexity, n_iter, lr, top_n=False, reverse=False, strange_scaling=False, is_coarse=True):
    data = prepare_data(query_matcher, top_n=top_n, reverse=reverse, strange_scaling=strange_scaling, is_coarse=is_coarse)
    pca_data = pd.DataFrame(PCA(n_components=50).fit_transform(data), index=data.index)
    names_n_labels = [[name, query_matcher.map_to_label(name, is_coarse)] for name in list(pca_data.index)]
    tsne_result_pca = compute_tsne(pca_data, perplexity, n_iter, lr, names_n_labels, is_coarse=True, d=2, num_jobs=None)
    return tsne_result_pca


class TsneVisualiser:
    def __init__(self, query_matcher, labels, filename, recompute=True, is_coarse=False):
        self.query_matcher = query_matcher
        self.labels = labels
        # self.values = values
        self.filename_csv = f"figs/{filename}.csv"
        self.filename_html = f"figs/{filename}.html"
        self.recompute = recompute
        self.is_coarse = is_coarse

    def plot(self):
        # If the file exists just show and return
        if self.file_exist():
            webbrowser.open('file://' + os.path.realpath(self.filename_html))
            return

        # Calculate perplexity
        if self.recompute:
            df = compute_tsne_subset(self.query_matcher, 19, int(10e6), 1, 15, is_coarse=False)
            df.to_csv(self.filename_csv)
        df = pd.read_csv(self.filename_csv, index_col=0)

        # Plotting, saving and displaying
        unique_classes = sorted(list(set(df['label'])))
        classification_indexes = [unique_classes.index(x) for x in df['label']]
        colors = rand_cmap(len(unique_classes), return_hex=True)  # cc.b_glasbey_bw[0:len(unique_classes)]
        draw_colors = [colors[classification_indexes[x]] for x in range(df.shape[0])]
        TOOLS = [
            "pan", "wheel_zoom", "zoom_in", "zoom_out", "box_zoom", "undo", "redo", "reset", "tap", "save", "box_select", "poly_select", "lasso_select",
            HoverTool(tooltips='@label')
        ]
        source = ColumnDataSource(data={'x': df["x"], 'y': df["y"], 'label': df["label"], 'colors': draw_colors})
        p = figure(title='tSNE', x_axis_label='tSNE 1', y_axis_label='tSNE 2', tools=TOOLS, plot_width=900, plot_height=900)
        p.circle(x='x', y='y', source=source, color="colors", alpha=255)
        output_file(self.filename_html, title="tSNE Hell.")
        save(p)
        webbrowser.open('file://' + os.path.realpath(self.filename_html))

    def file_exist(self):
        if os.path.isfile(self.filename_csv):
            if self.recompute:
                os.remove(self.filename_csv)
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