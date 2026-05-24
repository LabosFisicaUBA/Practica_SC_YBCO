"""
main
====

Entry-point script for the experimental-data analysis workflow.

This module coordinates the current exploratory stage of the superconductivity
data-analysis pipeline. It reads external configuration parameters, moves to the
raw-data directory, loads selected experimental datasets, and generates basic
Matplotlib figures for quick inspection.

Current workflow
----------------
1. Read the ``parameters.ini`` configuration file.
2. Move to the directory containing the raw experimental data files.
3. Load susceptibility and magnetization datasets using
   :func:`processing.data_proc`.
4. Print numerical parameters from the configuration file.
5. Generate exploratory plots using :func:`graphics.create_figure`.
6. Display the generated Matplotlib figures.

Notes
-----
This file is expected to change frequently while the analysis workflow is being
developed. For that reason, the documentation is intentionally light and focused
on the current executable logic.

Stable data-processing and plotting routines should be documented in their own
modules. Experiment-specific choices, such as which files are loaded and which
columns are plotted, are intentionally kept in this script for now.
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
    Read the configuration file and move to the data directory.

    This function performs the basic initialization required by the exploratory
    analysis script. It first reads the ``parameters.ini`` file from the current
    working directory and then changes the process working directory to the
    directory containing the raw data files.

    Parameters
    ----------
    _dir : str, optional
        Path to the directory containing the experimental data files.
        The default is ``"../datos"``.

    Returns
    -------
    config : configparser.ConfigParser
        Parsed configuration object containing the parameters read from
        ``parameters.ini``.

    Raises
    ------
    FileNotFoundError
        Raised if ``parameters.ini`` or the requested data directory does not
        exist.
    OSError
        Raised if ``parameters.ini`` cannot be opened or if the working
        directory cannot be changed.

    Notes
    -----
    The configuration file is read before changing directory. Therefore,
    ``parameters.ini`` is expected to be located in the directory from which the
    script is executed, not necessarily inside the data directory.

    This function changes the global current working directory of the Python
    process using :func:`os.chdir`. This is convenient for quick exploratory
    scripts, but in larger workflows it is usually safer to build explicit file
    paths instead of changing process state.
    """

    # Read the parameters configuration file.
    config = configparser.ConfigParser()

    try:
        with open("parameters.ini", "r") as params_file:
            config.read_file(params_file)

    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        raise FileNotFoundError(
            "parameters.ini file not found"
        ) from e

    except OSError as e:
        print(e, file=sys.stderr)
        raise OSError(
            "Could not open parameters.ini"
        ) from e

    # Move to the directory where the raw data files are stored.
    try:
        os.chdir(_dir)

    except FileNotFoundError as e:
        raise FileNotFoundError(
            f"Directory not found: {_dir!r}"
        ) from e

    except OSError as e:
        print(e, file=sys.stderr)
        raise OSError(
            f"Could not change directory: {_dir!r}"
        ) from e

    return config


def print_data(labels, data):
    """
    Print column labels and numerical data to standard output.

    Parameters
    ----------
    labels : list of str
        Column labels associated with the numerical data arrays.
    data : list of numpy.ndarray
        Column-wise numerical data. Each entry in ``data`` is assumed to
        correspond to the label with the same index in ``labels``.

    Returns
    -------
    None

    Notes
    -----
    This function is intended for quick inspection and debugging. It prints each
    column separately, together with the row index of every value.

    For large experimental datasets, this function can produce very verbose
    output and should not be used as the main reporting mechanism.
    """

    for idx, label in enumerate(labels):
        print(label)

        for d_idx, val in enumerate(data[idx]):
            print("{0}\t{1}".format(d_idx, val))

        print("\n")


def main():
    """
    Run the current experimental-data analysis workflow.

    The function initializes the workflow, reads numerical parameters from the
    configuration file, loads the selected experimental datasets, creates basic
    exploratory figures, and displays them using Matplotlib.

    The currently loaded datasets are:

    - AC magnetic susceptibility as a function of temperature.
    - Detailed magnetization loops at 10 K, 40 K, and 70 K.
    - Magnetization loops measured at several temperatures and fields.

    Returns
    -------
    int
        Exit status code. Returns ``0`` if the workflow finishes successfully
        and ``-1`` if an exception is caught during execution.

    Raises
    ------
    None
        Exceptions raised by lower-level functions are caught inside this
        function and reported to standard output.

    Notes
    -----
    This function is intentionally script-like. The selected filenames and plotted
    columns are currently hard-coded because the analysis stage is still
    exploratory.

    Once the workflow stabilizes, filename selection, plotted columns, titles,
    and output options should be moved to ``parameters.ini`` or to a dedicated
    configuration layer.
    """
    
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
        
        fig1, ax1 = create_figure(labels, data, title)
        fig2, ax2 = create_figure(labels, data, title, ycol_num = 2)

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