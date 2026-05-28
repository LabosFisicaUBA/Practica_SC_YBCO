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

import numpy as np
from matplotlib.collections import LineCollection
import warnings

def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)