"""
This module contains functions to compute classic ratios from common glottal signals

The definitions of these ratios come from a variety of papers including
E. Holmberg, R. Hillman, J. Perkell - Glottal airflow ... - 1998 - JASA
"""

import numpy as np

# Returns arrays with the same shape as the signal
def is_closed(t, y, closed_ub):
    """
    Return a float array indicating closure
    """
    return y < closed_ub

def is_open(t, y, closed_ub):
    return np.logical_not(is_closed(t, y, closed_ub))

def is_closing(t, y, closed_ub):
    y_prime = np.gradient(y, t)
    return np.logical_and(is_open(t, y, closed_ub), y_prime < 0)

def is_opening(t, y, closed_ub):
    y_prime = np.gradient(y, t)
    return np.logical_and(is_open(t, y, closed_ub), y_prime >= 0)

# Returns scalar summaries of the signal
def closed_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    ind_closed = np.array(is_closed(t, y, closed_ub), dtype=np.float)
    closed_duration = np.trapz(ind_closed, x=t)
    duration = t[-1]-t[0]
    return closed_duration/duration

def open_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of open time to total time over the provided signal
    """
    return 1 - closed_ratio(t, y, closed_ub)

def closing_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    ind_closing = np.array(is_closing(t, y, closed_ub), dtype=np.float)
    closing_duration = np.trapz(ind_closing, x=t)
    duration = t[-1]-t[0]
    return closing_duration/duration

def opening_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    ind_opening = np.array(is_opening(t, y, closed_ub), dtype=np.float)
    opening_duration = np.trapz(ind_opening, x=t)
    duration = t[-1]-t[0]
    return opening_duration/duration

def speed_ratio(t, y, closed_ub=0.0):
    return opening_ratio(t, y, closed_ub)/closing_ratio(t, y, closed_ub)

def mfdr(t, y):
    """
    Return the maximum flow declination rate (MFDR)
    """
    yp = np.gradient(y, t)
    return np.min(yp)

def ac_flow(t, y):
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
