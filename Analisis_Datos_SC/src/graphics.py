"""
graphics
========

Plotting utilities for experimental data analysis.

This module provides helper functions to generate Matplotlib figures from
column-wise numerical data loaded by the data-processing pipeline.

The expected data structure is the one returned by ``processing.data_proc``:

- ``labels``: list of column labels.
- ``data``: list of one-dimensional NumPy arrays, one array per column.

Notes
-----
This module is intended for quick exploratory visualization of experimental
datasets. More specialized plotting routines can be added later as the analysis
workflow becomes more stable.
"""

import matplotlib.pyplot as plt


def create_figure(labels, data, title, xcol_num=0, ycol_num=1, linestyle='-', linewidth=2, marker=None, markersize=3):
    """
    Create a two-dimensional line plot from selected data columns.

    This function plots one selected column of the dataset against another.
    The column labels are used automatically as axis labels, and the selected
    y-column label is also used as the legend entry.

    Parameters
    ----------
    labels : list of str
        Column labels associated with the numerical data.
    data : list of array-like
        Column-wise numerical data. Each entry in ``data`` must correspond
        to the label with the same index in ``labels``.
    title : str
        Title of the plot.
    xcol_num : int, optional
        Index of the column to use as the horizontal axis. The default is ``0``.
    ycol_num : int, optional
        Index of the column to use as the vertical axis. The default is ``1``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object created by Matplotlib.
    ax : matplotlib.axes.Axes
        Axes object containing the plotted curve.

    Raises
    ------
    IndexError
        Raised if ``xcol_num`` or ``ycol_num`` is outside the valid range of
        available columns.
    ValueError
        Raised by Matplotlib if the selected x and y data arrays do not have
        compatible dimensions.

    Notes
    -----
    The function creates a Matplotlib figure and axes internally, but does not
    call ``plt.show()``. This allows the caller to decide when and how to display
    or save the figure.
    """

    # Plot the selected data columns.
    fig, ax = plt.subplots()

    x_label = labels[xcol_num]
    y_label = labels[ycol_num]

    x_data = data[xcol_num]
    y_data = data[ycol_num]

    plt.title(title)

    curve = ax.plot(x_data, y_data, linestyle=linestyle, linewidth=linewidth, marker=marker, markersize=markersize, label=y_label )

    fig.legend(curve, [y_label])

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    
    return fig, ax

def create_scatter_figure(labels, data, title, xcol_num=0, ycol_num=1, marker='s', markersize=4 ):
    """
    Create a two-dimensional scatter plot from selected data columns.

    This function plots one selected column of the dataset against another.
    The column labels are used automatically as axis labels, and the selected
    y-column label is also used as the legend entry.

    Parameters
    ----------
    labels : list of str
        Column labels associated with the numerical data.
    data : list of array-like
        Column-wise numerical data. Each entry in ``data`` must correspond
        to the label with the same index in ``labels``.
    title : str
        Title of the plot.
    xcol_num : int, optional
        Index of the column to use as the horizontal axis. The default is ``0``.
    ycol_num : int, optional
        Index of the column to use as the vertical axis. The default is ``1``.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Figure object created by Matplotlib.
    ax : matplotlib.axes.Axes
        Axes object containing the plotted curve.

    Raises
    ------
    IndexError
        Raised if ``xcol_num`` or ``ycol_num`` is outside the valid range of
        available columns.
    ValueError
        Raised by Matplotlib if the selected x and y data arrays do not have
        compatible dimensions.

    Notes
    -----
    The function creates a Matplotlib figure and axes internally, but does not
    call ``plt.show()``. This allows the caller to decide when and how to display
    or save the figure.
    """

    # Plot the selected data columns.
    fig, ax = plt.subplots()

    x_label = labels[xcol_num]
    y_label = labels[ycol_num]

    x_data = data[xcol_num]
    y_data = data[ycol_num]

    plt.title(title)

    
    curve = ax.scatter(x_data, y_data, marker=marker, s=markersize, label=y_label )
    
    fig.legend()

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(True)
    
    return fig, ax