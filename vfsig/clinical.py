"""
This module contains functions to compute classic ratios from common vocal fold
(VF) derived signals

The definitions of these ratios come from a variety of papers including
E. Holmberg, R. Hillman, J. Perkell - Glottal airflow ... - 1998 - JASA
"""

import numpy as np

## Indicating functions
# These functions indicate whether the VFs are in some state or not
# (open, closed, etc..)
def is_closed(t, y, closed_ub):
    """
    Return a boolean array indicating VF closure
    """
    return y < closed_ub

def is_open(t, y, closed_ub):
    """
    Return a boolean array indicating VFs are open
    """
    return np.logical_not(is_closed(t, y, closed_ub))

def is_closing(t, y, closed_ub):
    """
    Return a boolean array indicating VFs are closing
    """
    y_prime = np.gradient(y, t)
    return np.logical_and(is_open(t, y, closed_ub), y_prime < 0)

def is_opening(t, y, closed_ub):
    """
    Return a boolean array indicating VFs are opening
    """
    y_prime = np.gradient(y, t)
    return np.logical_and(is_open(t, y, closed_ub), y_prime >= 0)

# Return scalar summaries of the signal
def closed_ratio(t, y, closed_ub=0.0):
    """
    Return the closed ratio

    This is the ratio of time spent closed to total time
    """
    ind_closed = np.array(is_closed(t, y, closed_ub), dtype=np.float)
    closed_duration = np.trapz(ind_closed, x=t)
    duration = t[-1]-t[0]
    return closed_duration/duration

def open_ratio(t, y, closed_ub=0.0):
    """
    Return the open ratio

    This is the ratio of time spent open to total time
    """
    return 1 - closed_ratio(t, y, closed_ub)

def closing_ratio(t, y, closed_ub=0.0):
    """
    Return the closing ratio

    This is the ratio of time spent closing to the total time
    """
    ind_closing = np.array(is_closing(t, y, closed_ub), dtype=np.float)
    closing_duration = np.trapz(ind_closing, x=t)
    duration = t[-1]-t[0]
    return closing_duration/duration

def opening_ratio(t, y, closed_ub=0.0):
    """
    Return the opening ratio

    This is the ratio of time spent opening to the total time
    """
    ind_opening = np.array(is_opening(t, y, closed_ub), dtype=np.float)
    opening_duration = np.trapz(ind_opening, x=t)
    duration = t[-1]-t[0]
    return opening_duration/duration

def speed_ratio(t, y, closed_ub=0.0):
    """
    Return the speed ratio

    This is the ratio of opening to closing times
    """
    return opening_ratio(t, y, closed_ub)/closing_ratio(t, y, closed_ub)

def mfdr(t, y):
    """
    Return the maximum flow declination rate (MFDR)
    """
    yp = np.gradient(y, t)
    return np.min(yp)

def ac_flow(t, y):
    """
    Return the AC flow

    This is the amplitude from minimum to maximum of the signal
    """
    return np.max(y) - np.min(y)

def acdc(t, y):
    """
    See Holmberg et al. for the definition
    """
    y_ac = y - y.min()

    T = t[-1]-t[0]
    y_ac_rms = np.sqrt(np.trapz(t, y_ac**2)/T)
    y_ac_mean = np.trapz(t, y_ac)/T
    return y_ac_rms/y_ac_mean
