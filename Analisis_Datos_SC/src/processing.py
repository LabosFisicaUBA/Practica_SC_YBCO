"""
processing
==========

Utilities for loading tab-separated numerical data files.

This module provides a simple parser for experimental data files whose first
line contains column labels and whose remaining lines contain numerical values.
The data are returned column-wise as NumPy arrays, preserving the order of the
input file.

The expected input format is::

    label_1<TAB>label_2<TAB>...<TAB>label_n
    value_11<TAB>value_12<TAB>...<TAB>value_1n
    value_21<TAB>value_22<TAB>...<TAB>value_2n
    ...

Notes
-----
The parser assumes that every data row has the same number of tab-separated
entries as the header row. All data entries are converted to double-precision
floating-point values.
"""

import numpy as np
from numpy import double
import sys

def data_proc(archivo):
    """
    Load a tab-separated numerical data file.

    The first line of the file is interpreted as the list of column labels.
    Every subsequent line is interpreted as a row of numerical data. The output
    data are organized by columns, so that each entry in the returned ``data``
    list corresponds to one column of the original file.

    Parameters
    ----------
    archivo : str or path-like
        Path to the tab-separated data file.

    Returns
    -------
    labels : list of str
        Column labels read from the first line of the file.
    data : list of numpy.ndarray
        Column-wise numerical data. Each element of the list is a one-dimensional
        NumPy array with dtype compatible with double precision.

    Raises
    ------
    ValueError
        Raised if the file is empty, has inconsistent row lengths, or contains
        a value that cannot be converted to double precision.
    FileNotFoundError
        Raised if the input file does not exist.
    OSError
        Raised if the file exists but cannot be opened or read.

    Notes
    -----
    The function assumes a rectangular data table: every numerical row must have
    the same number of tab-separated columns as the header row.
    """
    # Define data containers.
    labels = list()
    data = list()

    try :
        
        # Open the data file and read it line by line.
        with open(archivo, 'r') as fp:
            for index, line in enumerate(fp):
                # The first line contains the column labels.
                if index == 0:
                    labels = line.rstrip("\r\n").split('\t')
        
                    for idx, _ in enumerate(labels):
                        data.append(list())
        
                # Remaining lines contain the numerical data.
                else:
                    # Use index - 1 conceptually because the first line is the header.
                    d = line.rstrip("\r\n").split('\t')
                    # Checks the table size consistency compared to the columns labels
                    if len(d) != len(labels):
                        raise ValueError(
                            f"Invalid row length in file '{archivo}', "
                            f"line {index + 1}: expected {len(labels)} columns, got {len(d)}"
                        )
                    for idx, val in enumerate(d):
                        try:
                            val = double(val)
                        except ValueError as exc:
                            raise ValueError(
                                f"Invalid numerical value in file '{archivo}', "
                                f"line {index + 1}, column {idx + 1}: {val!r}"
                            ) from exc
                        data[idx].append(val)
        
            # Convert each data column into a NumPy array.
            for idx, col in enumerate(data):
                data[idx] = np.array(col)
                
    except OSError as e:
        print(e, file=sys.stderr)
        raise OSError(
            f"Could not read data file: {archivo!r}"
        ) from e
        
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Data file not found: {archivo!r}"
        ) from e
    
    return labels, data