"""
main
====

Entry-point script for the experimental-data analysis workflow.

This module coordinates the first stage of the analysis pipeline:

1. Move to the directory containing the raw data files.
2. Select one experimental data file.
3. Load the file using :func:`processing.data_proc`.
4. Print the loaded columns for inspection.

Notes
-----
This file is expected to change frequently while the analysis workflow is
being developed. For that reason, the documentation is intentionally light:
stable helper functions are documented, while experiment-specific logic should
remain easy to modify.
"""

import os
import sys
from processing import data_proc


def setup(_dir="../datos"):
    """
    Move the current working directory to the data directory.

    Parameters
    ----------
    _dir : str, optional
        Path to the directory containing the experimental data files.
        The default is ``"../datos"``.

    Returns
    -------
    None

    Notes
    -----
    This function changes the global current working directory of the Python
    process using :func:`os.chdir`. This is convenient for quick exploratory
    scripts, but in larger workflows it is usually safer to build explicit file
    paths instead of changing the process state.
    """

    # Move to the directory where the raw data files are stored.
    try:
        os.chdir(_dir)
    except OSError as e:
        print(e, file=sys.stderr)
        raise OSError(
            f"Could not change directory: {dir!r}"
        ) from e
        
    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Directory not found: {dir!r}"
        ) from e



def print_data(labels, data):
    """
    Print column labels and numerical data to standard output.

    Parameters
    ----------
    labels : list of str
        Column labels associated with the data arrays.
    data : list of numpy.ndarray
        Column-wise numerical data. Each entry in ``data`` is assumed to
        correspond to the label with the same index in ``labels``.

    Returns
    -------
    None

    Notes
    -----
    This function is intended for quick inspection/debugging. It prints each
    column separately, together with the row index of every value.
    """

    for idx, label in enumerate(labels):
        print(label)

        for d_idx, val in enumerate(data[idx]):
            print("{0}\t{1}".format(d_idx, val))

        print("\n")


def main():
    # Retrieve the Command Line parameters
    argv = sys.argv
    
    # Prepare the working directory.
    

    try:
        
        setup()
         
        # Name of the experimental data file to process.
        archivo = "susceptibilidad_alterna_Hdc_0Oe_Hac_1Oe_f_1kHz.txt"
        #archivo = "misdatos.txt"
    
        # Load and inspect the data.
        labels, data = data_proc(archivo)
    except Exception as e:
        return -1
    
    print_data(labels, data)
    
    return 0

if __name__ == '__main__':
    exitcode = main()
    #print(exitcode)
    sys.exit(exitcode)