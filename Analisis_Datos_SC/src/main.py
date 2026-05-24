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
from graphics import create_figure
import matplotlib.pyplot as plt
import configparser
import numpy as np 


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

    # Read the parameters configuration file
    config = configparser.ConfigParser()
    try:
        with open('parameters.ini', 'r' ) as params_file:
            config.read_file(params_file)           
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise FileNotFoundError(
            f"parameters.ini file not found"
        ) from e
    except OSError as e:
        print(e, file=sys.stderr)
        raise OSError(
            f"Could not open parameters.ini"
        ) from e
        
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
    
    return config


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
        
        config = setup()
        params = config['params']
        for key in params :
            print("param:  {0}\t value: {1}".format(key, np.double(params[key])))
         
        # Name of the experimental data file to process.
        chi = "susceptibilidad_alterna_Hdc_0Oe_Hac_1Oe_f_1kHz.txt"
        M_10K = "lazo_detallado_M(H)_10K.txt"
        M_40K = "lazo_detallado_M(H)_40K.txt"
        M_70K = "lazo_detallado_M(H)_70K.txt"
        M_H_T = "lazos_M_variasT_variasH.txt"
        
    
        # Load and inspect the data.
        l_chi, data_chi = data_proc(chi)
        l_M_10K, data_M_10K = data_proc(M_10K)
        l_M_40K, data_M_40K = data_proc(M_40K)
        l_M_70K, data_M_70K = data_proc(M_70K)
        l_M_H_T, data_M_H_T = data_proc(M_H_T)
        
        
        labels = l_chi 
        data = data_chi
        xcol_num = 0
        ycol_num = 1
        fig_num = 0
        title = "Magnetic Susceptibility vs T"
        
        create_figure(labels, data, title)
        create_figure(labels, data, title, ycol_num = 2)

        plt.show()
        
    except Exception as e:
        print(e)
        return -1
    
    #print_data(l_M_10K, data_M_10K)
    
    return 0

if __name__ == '__main__':
    exitcode = main()
    #print(exitcode)
    sys.exit(exitcode)