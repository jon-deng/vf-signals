"""
This module contains functions to compute frequency domain quantities of a signal
"""
import numpy as np
from numpy import fft
import scipy.interpolate as interpolate

def estimate_fundamental_mode(y, dt: float=1, flb: float=0, fub: float=np.inf, axis: int=-1):
    """
    Return information about the fundamental mode of a signal

    Returns
    -------
    f0 : float
        Fundamental frequency estimate
    phi0 : float
        Phase of fundamental mode
    """
    # Remove the mean component of y
    y_ = y - np.mean(y, axis=axis, keepdims=True)

    # Compute the DFT and frequency components
    N = y_.shape[axis]
    dfty = fft.rfft(y_, N, axis=axis)
    f = fft.rfftfreq(N, d=dt)
    df = f[1] - f[0]

     # Offset `f`'s shape so that it broadcasts with `dfty`
    if axis < 0:
        noffset = -1 - axis
    else:
        noffset = y.ndim - axis - 1
    _idx_inrange = np.logical_and(f>flb, f<fub)
    idx_inrange = (Ellipsis,) + (_idx_inrange,) + (slice(None),)*noffset

    f = f[_idx_inrange]
    dfty = dfty[idx_inrange]

    _idx_f0 = np.argmax(np.abs(dfty), axis=axis)
    idx_f0 = (Ellipsis,) + (_idx_f0,) + (slice(None),)*noffset

    return f[_idx_f0], np.angle(dfty[idx_f0]), df

def estimate_periodic_statistics(y, n_period):
    """
    Return statistics on the periodic behaviour of a signal

    Parameters
    ----------
    y : array_like
        The signal to compute statistics on
    n_period : int
        The number of periods the signal is composed of. This can be computed
        from fourier analysis (see `estimate_fundamental_mode`).

    Returns
    -------
    f0 : float
        Fundamental frequency estimate
    phi0 : float
        Phase of fundamental mode
    """
    # Only accept 1D arrays
    assert len(y.shape) == 1

    # Compute a number of points per period to segment the signal with
    n_sample = y.size
    n_per_period = int(round(n_sample/n_period))

    # Resample the signal so that it has `n_per_period` points in each period
    n_resample = n_per_period * n_period
    re_y = np.interp(np.linspace(0, 1, n_resample), np.linspace(0, 1, n_sample), y)

    assert re_y.size == n_resample

    # Stack each period on top of the other so that means and standard deviations
    # can be taken over multiple periods
    y_periods = np.reshape(re_y, (n_period, -1))
    assert y_periods.shape[-1] == n_per_period

    # Compute the mean and standard deviation over periods
    mean_y_period = np.mean(y_periods, axis=0)
    stdev_y_period = np.std(y_periods, axis=0)

    return mean_y_period, stdev_y_period


def modal_info(t, y):
    """
    Return information about the modes of a signal

    Returns
    -------
    f0 : float
        Fundamental frequency estimate
    phi0 : float
        Phase of fundamental frequency
    """
    # Remove the mean component of y
    y_ = y - y.mean()

    # Compute fft and freq
    dfty = fft.rfft(y_)
    f = fft.rfftfreq(t.size, d=(t[-1]-t[0])/(t.size-1))

    idx_f0 = np.argmax(np.abs(dfty))

    return f[idx_f0], np.angle(dfty[idx_f0])

def fft_decomposition(dt, y):
    """
    Decompose a real signal (t, y) into the frequency domain as (f, yhat)
    """
    # Compute fft and freq
    yy = fft.rfft(y, n=y.size)
    freq = fft.rfftfreq(yy.size, d=dt)

    return freq, yy


    N = u.size
    return np.sum(psd_from_rfft(u, v, axis))