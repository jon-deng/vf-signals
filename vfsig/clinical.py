"""
This module contains functions to compute classic ratios from common glottal signals

The definitions of these ratios come from a variety of papers including
E. Holmberg, R. Hillman, J. Perkell - Glottal airflow ... - 1998 - JASA
"""

import numpy as np

# Returns arrays with the same shape as the signal
def closed(t, y, closed_ub):
    """
    Return a boolean array indicating closure
    """
    return y < closed_ub

def open(t, y, closed_ub):
    return not closed(t, y, closed_ub)

def closing(t, y, closed_ub):
    y_prime = np.gradient(y, t)
    return np.logical_and(y > closed_ub, y_prime < 0)

def opening(t, y, closed_ub):
    y_prime = np.gradient(y, t)
    return np.logical_and(y > closed_ub, y_prime > 0)

# Returns scalar summaries of the signal
def closed_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    y_closed = closed(t, y, closed_ub)
    closed_duration = np.trapz(y_closed, x=t)
    duration = t[-1]-t[0]
    return closed_duration/duration

def open_ratio(t, y, open_lb=0.0):
    """
    Return the ratio of open time to total time over the provided signal
    """
    return 1 - closed_ratio(t, y, closed_ub=open_lb)

def closing_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    y_closing = closing(t, y, closed_ub)
    closing_duration = np.trapz(y_closing, x=t)
    duration = t[-1]-t[0]
    return closing_duration/duration

def opening_ratio(t, y, closed_ub=0.0):
    """
    Return the ratio of closed time to total time over the provided signal
    """
    y_opening = opening(t, y, closed_ub)
    opening_duration = np.trapz(y_opening, x=t)
    duration = t[-1]-t[0]
    return opening_duration/duration

def speed_ratio(t, y):
    pass

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
