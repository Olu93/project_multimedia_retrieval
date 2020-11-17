from typing import OrderedDict

from helper.misc import load_feature_data
from helper.config import FEATURE_DATA_FILE
from collections import Counter
from matplotlib import pyplot as plt
import pandas as pd


# Text Wrapping
# Defines wrapText which will attach an event to a given mpl.text object,
# wrapping it within the parent axes object.  Also defines a the convenience
# function textBox() which effectively converts an axes to a text box.
# https://stackoverflow.com/a/33790466/4162265
def wrapText(text, ax, margin=4):
    """ Attaches an on-draw event to a given mpl.text object which will
        automatically wrap its string wthin the parent axes object.

        The margin argument controls the gap between the text and axes frame
        in points.
    """
    # ax = text.get_axes()
    margin = margin / 72 * ax.figure.get_dpi()

    def _wrap(event):
        """Wraps text within its parent axes."""
        def _width(s):
            """Gets the length of a string in pixels."""
            text.set_text(s)
            return text.get_window_extent().width

        # Find available space
        clip = ax.get_window_extent()
        x0, y0 = text.get_transform().transform(text.get_position())
        if text.get_horizontalalignment() == 'left':
            width = clip.x1 - x0 - margin
        elif text.get_horizontalalignment() == 'right':
            width = x0 - clip.x0 - margin
        else:
            width = (min(clip.x1 - x0, x0 - clip.x0) - margin) * 2

        # Wrap the text string
        words = [''] + _splitText(text.get_text())[::-1]
        wrapped = []

        line = words.pop()
        while words:
            line = line if line else words.pop()
            lastLine = line

            while _width(line) <= width:
                if words:
                    lastLine = line
                    line += words.pop()
                    # Add in any whitespace since it will not affect redraw width
                    while words and (words[-1].strip() == ''):
                        line += words.pop()
                else:
                    lastLine = line
                    break

            wrapped.append(lastLine)
            line = line[len(lastLine):]
            if not words and line:
                wrapped.append(line)

        text.set_text('\n'.join(wrapped))

        # Draw wrapped string after disabling events to prevent recursion
        handles = ax.figure.canvas.callbacks.callbacks[event.name]
        ax.figure.canvas.callbacks.callbacks[event.name] = {}
        ax.figure.canvas.draw()
        ax.figure.canvas.callbacks.callbacks[event.name] = handles

    ax.figure.canvas.mpl_connect('draw_event', _wrap)


def _splitText(text):
    """ Splits a string into its underlying chucks for wordwrapping.  This
        mostly relies on the textwrap library but has some additional logic to
        avoid splitting latex/mathtext segments.
    """
    import textwrap
    import re
    math_re = re.compile(r'(?<!\\)\$')
    textWrapper = textwrap.TextWrapper()

    if len(math_re.findall(text)) <= 1:
        return textWrapper._split(text)
    else:
        chunks = []
        for n, segment in enumerate(math_re.split(text)):
            if segment and (n % 2):
                # Mathtext
                chunks.append('${}$'.format(segment))
            else:
                chunks += textWrapper._split(segment)
        return chunks


def textBox(text, axes, ha='left', fontsize=12, margin=None, frame=True, **kwargs):
    """ Converts an axes to a text box by removing its ticks and creating a
        wrapped annotation.
    """
    if margin is None:
        margin = 6 if frame else 0
    axes.set_xticks([])
    axes.set_yticks([])
    axes.set_frame_on(frame)

    an = axes.annotate(text,
                       fontsize=fontsize,
                       xy=({
                           'left': 0,
                           'right': 1,
                           'center': 0.5
                       }[ha], 1),
                       ha=ha,
                       va='top',
                       xytext=(margin, -margin),
                       xycoords='axes fraction',
                       textcoords='offset points',
                       **kwargs)
    wrapText(an, margin=margin)
    return an


if __name__ == '__main__':
    df = pd.DataFrame(load_feature_data(FEATURE_DATA_FILE))
    co = pd.DataFrame(Counter(df["label"]).most_common()[::-1], columns=["label", "cnt"])
    cut_off = 5
    co_xl = co[co.cnt > cut_off]
    co_xs = co[co.cnt <= cut_off]
    fs = (12, 4)
    fig = plt.figure(constrained_layout=True, figsize=fs)
    gs = fig.add_gridspec(1, 3)

    ax1 = fig.add_subplot(gs[0, :2])

    ax1.set_xlabel("Classes")
    ax1.set_ylabel("Number of instances")
    ax1.set_xticklabels(co_xl.label, rotation=90)
    ax1.set_yticklabels(range(0, len(co_xl), 5))
    ax1.bar(co_xl.label, co_xl.cnt, align='edge')

    grouped = co_xs.groupby("cnt").label.apply(', '.join).reset_index()
    t1 = grouped.iloc[0, 1]
    t2 = grouped.iloc[1, 1]
    print(grouped)
    rows = list(grouped.label)
    text = list(grouped.cnt.astype(str))
    cols = ["Labels", "Cnt"]

    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis("off")
    an1 = ax2.annotate(t1, fontsize=10, xy=(0, 1), ha='left', va='top', xytext=(0, -6), xycoords='axes fraction', textcoords='offset points')
    an2 = ax2.annotate(t2, fontsize=10, xy=(0, .5), ha='left', va='top', xytext=(0, -6), xycoords='axes fraction', textcoords='offset points')
    wrapText(an1, ax2)
    wrapText(an2, ax2)

    # fig.text()
    # ax2.text(0, .5, grouped.iloc[0, 1], fontsize=12)
    # ax2.remove()
    # small_ones_removed = [key.replace("_", " ").title() for key, value in list(co.items()) if value > 4]
    # small_ones = [key.replace("_", " ").title() for key, value in list(co.items()) if value <= 4]
    fig.suptitle("Number of instance per class.")
    fig.tight_layout()
    # plt.savefig("figs\\instance_per_class.png")
    plt.show()
