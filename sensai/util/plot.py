from typing import Sequence

from matplotlib import pyplot as plt
import numpy as np


def plotMatrix(matrix, title, xticklabels: Sequence[str], yticklabels: Sequence[str], xlabel: str, ylabel: str, normalize=True):
    """
    :param matrix: matrix whose data to plot, where matrix[i, j] will be rendered at x=i, y=j
    :param title: the plot's title
    :param xticklabels: the labels for the x-axis ticks
    :param yticklabels: the labels for the y-axis ticks
    :param xlabel: the label for the x-axis
    :param ylabel: the label for the y-axis
    :param normalize: whether to normalise the matrix before plotting it (dividing each entry by the sum of all entries)
    :return: the plot's axes object
    """
    matrix = np.transpose(matrix)

    if normalize:
        matrix = matrix.astype('float') / matrix.sum()
    fig, ax = plt.subplots()
    fig.canvas.set_window_title(title)
    # We want to show all ticks...
    ax.set(xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=xticklabels, yticklabels=yticklabels,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel)
    im = ax.imshow(matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
        rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else ('.2f' if matrix.dtype == np.float else 'd')
    thresh = matrix.max() / 2.
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix[i, j], fmt),
                ha="center", va="center",
                color="white" if matrix[i, j] > thresh else "black")
    fig.tight_layout()
    return ax