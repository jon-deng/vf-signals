"""
This module contains functionality to compute frequency related quantities of a signal
"""

from typing import Optional, Mapping, Any, Tuple
from numpy.typing import NDArray

import warnings

import numpy as np
from numpy import fft
from scipy import signal

RealSignal = NDArray[float]
OptAxis = Optional[int]

FundamentalModeInfo = Tuple[float, float, float, float, Mapping[str, Any]]


def fundamental_mode_from_rfft(
    y: RealSignal,
    dt: Optional[float] = 1.0,
    axis: OptAxis = -1,
    freq_lb: Optional[float] = 0,
    freq_ub: Optional[float] = np.inf,
) -> FundamentalModeInfo:
    """
    Return information about the fundamental mode of a signal

    The fundamental mode is found based on the highest peak in a DFT.

    Parameters
    ----------
    y: RealSignal of shape (..., N)
        Time domain signal
    axis: OptAxis
        Axis to compute PSD along
    freq_lb, freq_ub: float
        Lower and upper bounds for the fundamental frequency estimate

    Returns
    -------
    freq : float
        Fundamental frequency estimate
    df : float
        The uncertainty in the fundamental frequency estimate (the size of a
        frequency bin from the DFT)
    phase : float
        Phase of fundamental mode in radians
    dphase : float
        The uncertainty in the phase estimate (currently neglected to be zero)
        TODO: This could be based on the phase difference between the DFT bin
        boundaries
    info : Mapping[str, Any]
    """
    # Remove the mean component of `y`; `y_mf` stands for mean-free
    y_mf = y - np.mean(y, axis=axis, keepdims=True)

    # Compute the DFT, frequency bins, and frequency spacing
    N = y_mf.shape[axis]
    dfty = fft.rfft(y_mf, N, axis=axis)
    freq = fft.rfftfreq(N, d=dt)
    df = freq[1] - freq[0]

    # Reshape `freq` so that it broadcasts with `dfty`
    # TODO: You can probably account for the `axis` argument with a decorator
    # that permutes axes such that the desired axis is moved to the last axis/dim
    if axis < 0:
        noffset = -1 - axis
    else:
        noffset = y.ndim - axis - 1
    _idx_inrange = np.logical_and(freq > freq_lb, freq < freq_ub)
    idx_inrange = (Ellipsis,) + (_idx_inrange,) + (slice(None),) * noffset

    freq = freq[_idx_inrange]
    dfty = dfty[idx_inrange]

    _idx_f0 = np.argmax(np.abs(dfty), axis=axis)
    idx_f0 = (Ellipsis,) + (_idx_f0,) + (slice(None),) * noffset

    info = {}
    dphase = 0
    return freq[_idx_f0], df, np.angle(dfty[idx_f0]), dphase, info


def fundamental_mode_from_peaks(
    y: RealSignal,
    dt: Optional[float] = 1.0,
    freq_lb: Optional[float] = 0,
    freq_ub: Optional[float] = np.inf,
    **find_peaks_kwargs
) -> FundamentalModeInfo:
    """
    Return information about the fundamental mode of a signal

    The fundamental mode is found based on measuring time intervals between
    signal peaks.

    Parameters
    ----------
    y: RealSignal of shape (N,)
        Time domain signal
    freq_lb, freq_ub: float
        Lower and upper bounds for the fundamental frequency estimate

    Returns
    -------
    freq : float
        Fundamental frequency estimate
    phi : float
        Phase of fundamental mode in radians
    df : float
        The uncertainty in the fundamental frequency estimate (the size of a
        frequency bin from the DFT)
    info : Mapping[str, Any]
    """

    idx_peaks, peak_properties = signal.find_peaks(y, **find_peaks_kwargs)

    if len(idx_peaks) >= 2:
        periods = dt * (idx_peaks[1:] - idx_peaks[:-1])
        mean_period = np.mean(periods)

        phases = dt * idx_peaks - mean_period * np.arange(idx_peaks.size)
        mean_phase = np.mean(phases)
        stdev_phase = np.std(phases)

        freqs = 1 / periods
        freqs = freqs[np.logical_and(freqs > freq_lb, freqs < freq_ub)]
        mean_freq = np.mean(freqs)
        stdev_freq = np.std(freqs)
    else:
        mean_freq, stdev_freq, mean_phase, stdev_phase = 4 * (np.nan,)
        if len(idx_peaks) == 1:
            warnings.warn("Found a single peak", RuntimeWarning)
        elif len(idx_peaks) == 0:
            warnings.warn("Found no peaks", RuntimeWarning)

    info = {'peaks': idx_peaks, 'peak_properties': peak_properties}
    return mean_freq, stdev_freq, mean_phase, stdev_phase, info


def periodic_stats(y, n_period):
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
    n_per_period = int(round(n_sample / n_period))

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
    f = fft.rfftfreq(t.size, d=(t[-1] - t[0]) / (t.size - 1))

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
