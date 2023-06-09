"""
This module contains functionality related to Fourier transforms
"""

from typing import Optional
from numpy.typing import NDArray

import numpy as np

ComplexSignal = NDArray[complex]
RealSignal = NDArray[float]
OptAxis = Optional[int]

def psd_from_fft(
        u: ComplexSignal, v: ComplexSignal,
        axis: OptAxis=-1
    ) -> ComplexSignal:
    """
    Return power spectral density (PSD) from Fourier domain signals

    If `u` and `v` have units of [a] and [b], respectively, then the PSD has
    units of [a][b]/[frequency bin]. To obtain physical units for the
    frequency bin, you will have to further divide the output of this function
    by the size of the frequency bin.
    i.e. if the output is `psd = psd_from_fft(u, v)` and the frequency bin size
    is `df` (for example, in [Hz]) `psd/df` has units of [a][b][Hz]^(-1).

    Parameters
    ----------
    u, v: ComplexSignal of shape (..., N)
        Frequency domain signals as obtained from `np.fft.fft`
    axis: OptAxis
        Axis to compute PSD along

    Returns
    -------
    psd: ComplexSignal of shape (..., N)
        The power spectral density
    """
    N = u.shape[axis]
    return 1/N * np.conjugate(u)*v

def power_from_fft(
        u: ComplexSignal, v: ComplexSignal,
        axis: OptAxis=-1
    ) -> ComplexSignal:
    """
    Return signal power from Fourier domain signals

    If `ut` and `vt` are corresponding time domain signals to `u` and `v`, respectively,
    the power returned is equivalent to `np.sum( np.conj(ut)*vt )`.

    Parameters
    ----------
    u, v : ComplexSignal of shape (..., N)
        Frequency domain signals as obtained from `np.fft.fft`
    axis : OptAxis
        Axis to compute power along

    Returns
    -------
    power: ComplexSignal of shape (..., )
        The signal power
    """
    return np.sum(psd_from_fft(u, v, axis), axis=axis)

def psd_from_rfft(
        u: ComplexSignal, v: ComplexSignal,
        axis: OptAxis=-1,
        n: Optional[int]=None
    ) -> RealSignal:
    """
    Return power spectral density (PSD) from one-sided Fourier domain signals

    If `u` and `v` have units of [a] and [b], respectively, then the PSD has
    units of [a][b]/[frequency bin]. To obtain physical units for the
    frequency bin, you will have to further divide the output of this function
    by the size of the frequency bin.
    i.e. if the output is `psd = psd_from_rfft(u, v)` and the frequency bin size
    is `df` (for example, in [Hz]) `psd/df` has units of [a][b][Hz]^(-1).

    Parameters
    ----------
    u, v : ComplexSignal of shape (..., N)
        Frequency domain signals as obtained from `np.fft.rfft`
    n : Optional[int]
        The number of points in the original untransformed signal. If not
        provided, the function assumes `n` is even which may lead to aliasing.
        See `np.fft.rfft` and `np.fft.rifft` for details on why you have to
        supply `n`.
    axis : OptAxis
        Axis to compute along

    Returns
    -------
    psd: RealSignal of shape (..., N)
        The power spectral density
    """
    if n is None:
        n = 2*u.shape[axis]

    NDIM = max(u.ndim, v.ndim)
    # Get a purely positive axis by accounting for negative `axis`
    if axis < 0:
        AXIS = NDIM+axis
    else:
        AXIS = axis
    N = u.shape[AXIS]

    # Create a scale array to account for dropped symmetric components in `np.fft.rfft`
    shape = (1,)*AXIS + (N,) + (1,)*(NDIM-AXIS-1)
    scale = np.ones(shape)

    if n//2 == 0:
        idx = (0,)*AXIS + (slice(1, None),) + (0,)*(NDIM-AXIS-1)
    else:
        idx = (0,)*AXIS + (slice(1, -1),) + (0,)*(NDIM-AXIS-1)
    scale[idx] = 2.0

    return 1/n * np.real(scale*np.conjugate(u)*v)

def power_from_rfft(
        u: ComplexSignal, v: ComplexSignal,
        axis: OptAxis=-1,
        n: Optional[int]=None
    ) -> RealSignal:
    """
    Return signal power from one-sided Fourier domain signals

    If `ut` and `vt` are the corresponding time domain signals, the power
    returned is equivalent to `np.sum(np.conj(ut)*vt)`.

    Parameters
    ----------
    u, v : ComplexSignal of shape (..., N)
        Frequency domain signals as obtained from `np.fft.rfft`
    n : int
        The number of points in the original untransformed signal. If not
        provided, the function assumes `n` is even which may lead to aliasing.
        See `np.fft.rfft` and `np.fft.rifft` for details on why you have to
        supply `n`.
    axis : OptAxis
        Axis to compute along

    Returns
    -------
    power: RealSignal of shape (..., )
        The signal power
    """
    return np.sum(psd_from_rfft(u, v, axis=axis, n=n), axis=axis)
