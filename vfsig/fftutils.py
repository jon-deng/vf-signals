"""
Contains functionality related to fourier transforms of data
"""

import numpy as np


def psd_from_fft(u, v, axis=-1):
    """
    Return power spectral density from fourier domain inputs

    Parameters
    ----------
    u, v : np.array
        Frequency domain signals
    axis : int
        Axis to compute along
    """
    N = u.size
    return 1/N * np.conjugate(u)*v

def power_from_fft(u, v, axis=-1):
    """
    Return power from fourier domain inputs

    If `ut` and `vt` are the corresponding time domain signals, the power
    returned is equivalent to `np.sum(np.conj(ut)*vt)`.

    Parameters
    ----------
    u, v : np.array
        Frequency domain signals
    axis : int
        Axis to compute along
    """
    return np.sum(psd_from_fft(u, v, axis))

def psd_from_rfft(u, v, n=None, axis=-1):
    """
    Return power spectral density from one-sided fourier domain inputs

    Parameters
    ----------
    u, v : np.array
        Frequency domain signals
    n : int
        The number of points in the original time-domain signal. If not
        provided, the function assumes `n` is even which may lead to aliasing.
        See `np.fft.rfft` and `np.fft.rifft` for details on why.
    axis : int
        Axis to compute along
    """
    if n is None:
        n = 2*u.shape[axis]

    NDIM = max(u.ndim, v.ndim)
    axis = NDIM+axis if axis < 0 else axis
    N = u.shape[axis]

    # Create scaling arrays to account for dropped symmetric components
    _shape = (0,)*(axis) + (N,) + (0,)*(NDIM-axis-1)
    _a = np.ones(_shape)

    if n//2 == 0:
        idx = (0,)*(axis) + (slice(1, None),) + (0,)*(NDIM-axis-1)
        _a[idx] = 2.0
    else:
        idx = (0,)*(axis) + (slice(1, -1),) + (0,)*(NDIM-axis-1)
        _a[idx] = 2.0
    breakpoint()

    return 1/n * np.real(_a*np.conjugate(u)*(v))

def power_from_rfft(u, v, n=None, axis=-1):
    """
    Return power from one-sided fourier domain inputs

    If `ut` and `vt` are the corresponding time domain signals, the power
    returned is equivalent to `np.sum(np.conj(ut)*vt)`.
    """
    return np.sum(psd_from_rfft(u, v, n=n, axis=axis))