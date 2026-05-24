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
from scipy.interpolate import PchipInterpolator
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

def split_hysteresis_branches(field, moment, tolerance=None):
    """
    Split a magnetic hysteresis loop into increasing-field and decreasing-field
    branches.

    Parameters
    ----------
    field : array-like
        Magnetic field values, in acquisition order.
    moment : array-like
        Magnetic moment values, in acquisition order.
    tolerance : float or None, optional
        Numerical tolerance used to classify field changes. If None, an automatic
        tolerance is estimated from the field step size.

    Returns
    -------
    up_branch : tuple of numpy.ndarray
        Tuple ``(H_up, m_up)`` containing points measured while the field
        increases.
    down_branch : tuple of numpy.ndarray
        Tuple ``(H_down, m_down)`` containing points measured while the field
        decreases.
    turning_points : tuple of numpy.ndarray
        Tuple ``(H_turn, m_turn)`` containing points near field turning regions
        where the sweep direction is ambiguous.

    Notes
    -----
    The input arrays must preserve the original acquisition order. Sorting the
    data by field before calling this function destroys the sweep-direction
    information.
    """
    field = np.asarray(field)
    moment = np.asarray(moment)

    if field.shape != moment.shape:
        raise ValueError(
            f"field and moment must have the same shape: "
            f"got {field.shape} and {moment.shape}."
        )

    dH = np.gradient(field)

    if tolerance is None:
        steps = np.abs(np.diff(field))
        steps = steps[steps > 0]

        if len(steps) == 0:
            raise ValueError("Cannot infer sweep direction from constant field data.")

        tolerance = 0.01 * np.median(steps)

    up_mask = dH > tolerance
    down_mask = dH < -tolerance
    turn_mask = ~(up_mask | down_mask)

    H_up = field[up_mask]
    m_up = moment[up_mask]

    H_down = field[down_mask]
    m_down = moment[down_mask]

    H_turn = field[turn_mask]
    m_turn = moment[turn_mask]

    return (H_up, m_up), (H_down, m_down), (H_turn, m_turn)

def zero_crossings_linear(x, y):
    """
    Find approximate zero crossings of y(x) by linear interpolation.
    """
    x = np.asarray(x)
    y = np.asarray(y)

    crossings = []

    for i in range(len(y) - 1):
        y0 = y[i]
        y1 = y[i + 1]

        if y0 == 0:
            crossings.append(x[i])

        elif y0 * y1 < 0:
            x0 = x[i]
            x1 = x[i + 1]

            x_cross = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(x_cross)

    return np.array(crossings)

def prepare_branch(H, m):
    """
    Sort a hysteresis branch by magnetic field and remove repeated field values.

    Parameters
    ----------
    H : array-like
        Magnetic field values.
    m : array-like
        Magnetic moment values.

    Returns
    -------
    H_sorted : numpy.ndarray
        Sorted magnetic field values.
    m_sorted : numpy.ndarray
        Magnetic moment values sorted consistently with ``H_sorted``.
    """
    H = np.asarray(H)
    m = np.asarray(m)

    order = np.argsort(H)
    H_sorted = H[order]
    m_sorted = m[order]

    # Remove duplicated H values by averaging m over equal-H groups.
    H_unique, inverse = np.unique(H_sorted, return_inverse=True)

    m_unique = np.zeros_like(H_unique, dtype=float)
    counts = np.zeros_like(H_unique, dtype=float)

    for idx, inv_idx in enumerate(inverse):
        m_unique[inv_idx] += m_sorted[idx]
        counts[inv_idx] += 1

    m_unique /= counts

    return H_unique, m_unique

def compute_delta_m(H_up, m_up, H_down, m_down, n_grid=1000):
    """
    Interpolate up and down hysteresis branches over a common field grid and
    compute their magnetic moment difference.

    Parameters
    ----------
    H_up : array-like
        Field values for the increasing-field branch.
    m_up : array-like
        Moment values for the increasing-field branch.
    H_down : array-like
        Field values for the decreasing-field branch.
    m_down : array-like
        Moment values for the decreasing-field branch.
    n_grid : int, optional
        Number of points in the common interpolation grid.

    Returns
    -------
    H_grid : numpy.ndarray
        Common magnetic field grid.
    m_up_grid : numpy.ndarray
        Interpolated increasing-branch moment.
    m_down_grid : numpy.ndarray
        Interpolated decreasing-branch moment.
    delta_m : numpy.ndarray
        Branch difference ``m_down_grid - m_up_grid``.
    """
    H_up, m_up = prepare_branch(H_up, m_up)
    H_down, m_down = prepare_branch(H_down, m_down)

    H_min = max(np.min(H_up), np.min(H_down))
    H_max = min(np.max(H_up), np.max(H_down))

    if H_min >= H_max:
        raise ValueError(
            "The up and down branches do not have an overlapping field range."
        )

    H_grid = np.linspace(H_min, H_max, n_grid)

    interp_up = PchipInterpolator(H_up, m_up)
    interp_down = PchipInterpolator(H_down, m_down)

    m_up_grid = interp_up(H_grid)
    m_down_grid = interp_down(H_grid)

    delta_m = m_down_grid - m_up_grid

    return H_grid, m_up_grid, m_down_grid, delta_m


def jc_bean_square(delta_m_emu, L_cm=0.1, thickness_cm=0.005):
    """
    Estimate J_c(H) using the extended Bean model for an approximately
    square platelet.

    Parameters
    ----------
    delta_m_emu : array-like
        Magnetic moment loop width, in emu.
    L_cm : float, optional
        In-plane side length of the approximately square sample, in cm.
        Default is 0.1 cm = 1 mm.
    thickness_cm : float, optional
        Sample thickness, in cm.
        Default is 0.005 cm = 0.05 mm.

    Returns
    -------
    Jc : numpy.ndarray
        Critical current density in A/cm^2.

    Notes
    -----
    This function assumes a square in-plane geometry, a = b = L, so that

        J_c = 30 Delta M / L,

    with Delta M = Delta m / V and V = L^2 thickness.
    """
    delta_m_emu = np.asarray(delta_m_emu, dtype=float)

    volume_cm3 = L_cm**2 * thickness_cm
    delta_M_emu_cm3 = delta_m_emu / volume_cm3

    Jc_A_cm2 = 30.0 * delta_M_emu_cm3 / L_cm

    return Jc_A_cm2

def jc_bean_rectangular(delta_m_emu, width_cm, length_cm, thickness_cm):
    """
    Estimate J_c(H) using the extended Bean model for a rectangular platelet.

    Parameters
    ----------
    delta_m_emu : array-like
        Magnetic moment loop width, in emu.
    width_cm : float
        First in-plane sample dimension, in cm.
    length_cm : float
        Second in-plane sample dimension, in cm.
    thickness_cm : float
        Sample thickness, in cm.

    Returns
    -------
    Jc : numpy.ndarray
        Critical current density in A/cm^2.
    """
    delta_m_emu = np.asarray(delta_m_emu, dtype=float)

    a = min(width_cm, length_cm)
    b = max(width_cm, length_cm)

    volume_cm3 = width_cm * length_cm * thickness_cm
    delta_M = delta_m_emu / volume_cm3

    geometric_factor = a * (1.0 - a / (3.0 * b))

    Jc = 20.0 * delta_M / geometric_factor

    return Jc

def jc_bean_square_demag(
    H_appl,
    m_up,
    m_down,
    width_cm=0.1,
    thickness_cm=0.005,
    n_grid=1000,
):
    """
    Compute Bean critical current density for an approximately square platelet,
    including a demagnetizing correction to the field axis.

    Parameters
    ----------
    H_appl : array-like
        Applied magnetic field grid, in Oe.
    m_up : array-like
        Increasing-field branch magnetic moment, in emu.
    m_down : array-like
        Decreasing-field branch magnetic moment, in emu.
    width_cm : float
        In-plane square side length, in cm.
    thickness_cm : float
        Plate thickness, in cm.
    n_grid : int
        Number of points in the common internal-field grid.

    Returns
    -------
    H_int_grid : numpy.ndarray
        Common internal magnetic field grid, in Oe.
    Jc : numpy.ndarray
        Critical current density, in A/cm^2.
    Nz : float
        Approximate out-of-plane demagnetizing factor.
    """
    H_appl = np.asarray(H_appl, dtype=float)
    m_up = np.asarray(m_up, dtype=float)
    m_down = np.asarray(m_down, dtype=float)

    volume_cm3 = width_cm**2 * thickness_cm

    M_up = m_up / volume_cm3
    M_down = m_down / volume_cm3

    Nz = 1.0 - 0.75 * thickness_cm / width_cm

    H_int_up = H_appl - 4.0 * np.pi * Nz * M_up
    H_int_down = H_appl - 4.0 * np.pi * Nz * M_down

    H_int_up, M_up = prepare_branch(H_int_up, M_up)
    H_int_down, M_down = prepare_branch(H_int_down, M_down)

    H_min = max(np.min(H_int_up), np.min(H_int_down))
    H_max = min(np.max(H_int_up), np.max(H_int_down))

    H_int_grid = np.linspace(H_min, H_max, n_grid)

    M_up_interp = PchipInterpolator(H_int_up, M_up)(H_int_grid)
    M_down_interp = PchipInterpolator(H_int_down, M_down)(H_int_grid)

    delta_M = M_down_interp - M_up_interp

    Jc = 30.0 * delta_M / width_cm

    return H_int_grid, Jc, Nz

def demag_corrected_field_average(
    H_appl,
    m_up,
    m_down,
    width_cm=0.1,
    thickness_cm=0.005,
):
    """
    Compute J_c for a square platelet and correct the field axis using the
    average magnetization of the two hysteresis branches.

    Parameters
    ----------
    H_appl : array-like
        Applied magnetic field, in Oe.
    m_up : array-like
        Increasing-field branch magnetic moment, in emu.
    m_down : array-like
        Decreasing-field branch magnetic moment, in emu.
    width_cm : float
        In-plane square side length, in cm.
    thickness_cm : float
        Plate thickness, in cm.

    Returns
    -------
    H_int : numpy.ndarray
        Approximate internal magnetic field, in Oe.
    Jc : numpy.ndarray
        Critical current density, in A/cm^2.
    Nz : float
        Approximate out-of-plane demagnetizing factor.
    """
    H_appl = np.asarray(H_appl, dtype=float)
    m_up = np.asarray(m_up, dtype=float)
    m_down = np.asarray(m_down, dtype=float)

    volume_cm3 = width_cm**2 * thickness_cm

    M_up = m_up / volume_cm3
    M_down = m_down / volume_cm3

    delta_M = M_down - M_up
    M_avg = 0.5 * (M_down + M_up)

    Nz = 1.0 - 0.75 * thickness_cm / width_cm

    H_int = H_appl - 4.0 * np.pi * Nz * M_avg

    Jc = 30.0 * delta_M / width_cm

    return H_int, Jc, Nz

def fwhm(x, y, peak_index=None, baseline=None):
    """
    Estimate the full width at half maximum of a single positive peak.

    Parameters
    ----------
    x : array-like
        Horizontal coordinate values.
    y : array-like
        Signal values.
    peak_index : int or None, optional
        Index of the peak maximum. If None, np.argmax(y) is used.
    baseline : float or None, optional
        Baseline value. If None, np.min(y) is used.

    Returns
    -------
    fwhm : float
        Full width at half maximum.
    x_left : float
        Left half-maximum crossing.
    x_right : float
        Right half-maximum crossing.
    half_height : float
        Half-maximum signal value.
    x_peak : float
        Peak position.
    y_peak : float
        Peak value.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape.")

    # Sort by x, useful if the data were not already ordered.
    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if peak_index is None:
        peak_index = np.argmax(y)
    else:
        # Convert original peak index into sorted-array index.
        peak_index = np.where(order == peak_index)[0][0]

    if baseline is None:
        baseline = np.min(y)

    x_peak = x[peak_index]
    y_peak = y[peak_index]

    half_height = baseline + 0.5 * (y_peak - baseline)

    # Left crossing.
    left_side = y[:peak_index + 1]
    left_candidates = np.where(left_side <= half_height)[0]

    if len(left_candidates) == 0:
        raise ValueError("Could not find left half-maximum crossing.")

    i_left = left_candidates[-1]

    if i_left == peak_index:
        raise ValueError("Invalid left crossing: peak is already below half height.")

    x0, x1 = x[i_left], x[i_left + 1]
    y0, y1 = y[i_left], y[i_left + 1]

    x_left = x0 + (half_height - y0) * (x1 - x0) / (y1 - y0)

    # Right crossing.
    right_side = y[peak_index:]
    right_candidates = np.where(right_side <= half_height)[0]

    if len(right_candidates) == 0:
        raise ValueError("Could not find right half-maximum crossing.")

    i_right = peak_index + right_candidates[0]

    if i_right == peak_index:
        raise ValueError("Invalid right crossing: peak is already below half height.")

    x0, x1 = x[i_right - 1], x[i_right]
    y0, y1 = y[i_right - 1], y[i_right]

    x_right = x0 + (half_height - y0) * (x1 - x0) / (y1 - y0)

    fwhm = x_right - x_left

    return fwhm, x_left, x_right, half_height, x_peak, y_peak