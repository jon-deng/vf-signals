"""
This module contains functions to compute classic ratios from common vocal fold
(VF) derived signals

The definitions of these ratios come from a variety of papers including
E. Holmberg, R. Hillman, J. Perkell - Glottal airflow ... - 1998 - JASA
"""

from xml.dom import ValidationErr
import numpy as np
import scipy as sp

## Indicating functions
# These functions indicate whether the VFs are in some state or not
# (open, closed, etc..)
def _add_signal_docstring(signal_function):
    add_docstring = """
    Parameters
    ----------
    y : array_like
        The glottal signal of interest
    t : array_like
        Time instances of the glottal signal. This should have a shape
        broadcastable to `y`
    dt : float
        The spacing between samples if sample spacing is uniform. If `t` is
        supplied, values of `dt` will be ignored
    closed_ub : float
        The value of `y` where for all `y < closed_ub`, the signal is assumed to
        indicate closure.

    Returns
    -------
    array_like
        Boolean array indicating whether the state is satisfied or not
    """
    signal_function.__doc__ = signal_function.__doc__ + add_docstring
    return signal_function

@_add_signal_docstring
def is_closed(y, t=None, dt=1.0, closed_ub=0):
    """
    Return a boolean array indicating VF closure

    """
    return y < closed_ub

@_add_signal_docstring
def is_open(y, t=None, dt=1.0, closed_ub=0):
    """
    Return a boolean array indicating VFs are open

    """
    return np.logical_not(
        is_closed(y, t=t, dt=dt, closed_ub=closed_ub)
        )

@_add_signal_docstring
def is_closing(y, t=None, dt=1.0, closed_ub=0):
    """
    Return a boolean array indicating VFs are closing

    """
    y_prime = np.gradient(y, t)
    return np.logical_and(
        is_open(y, t=t, dt=dt, closed_ub=closed_ub),
        y_prime < 0
        )

@_add_signal_docstring
def is_opening(y, t=None, dt=1.0, closed_ub=0):
    """
    Return a boolean array indicating VFs are opening

    """
    y_prime = np.gradient(y, t)
    return np.logical_and(
        is_open(y, t=t, dt=dt, closed_ub=closed_ub),
        y_prime >= 0
        )

## Return scalar summaries of the signal
def _add_measure_docstring(measure_function):
    add_docstring = """
    Parameters
    ----------
    y : array_like
        The glottal signal of interest
    t : array_like
        Time instances of the glottal signal. This should have a shape
        broadcastable to `y`
    dt : float
        The spacing between samples if sample spacing is uniform. If `t` is
        supplied, values of `dt` will be ignored
    axis : int
        The axis to reduce to a scalar measure. For example if `y` has a shape
        `(5, 1024)` indicating 1024 time samples, calling
        `closed_ratio(y, axis=-1)` will result in an array of shape `(5,)`
        indicating the closed ratio of each of the 5 sets of signals.
    closed_ub : float
        The value of `y` where for all `y < closed_ub`, the signal is assumed to
        indicate closure.

    Returns
    -------
    array_like
        An array containing the summary scalar. If `y` has a single axis, this
        is a float.
    """
    measure_function.__doc__ = measure_function.__doc__ + add_docstring
    return measure_function

@_add_measure_docstring
def closed_ratio(y, t=None, dt=1.0, axis=-1, closed_ub=0):
    """
    Return the closed ratio

    This is the ratio of time spent closed to total time

    """
    ind_kwargs = {
        't': t, 'dt': dt, 'closed_ub': closed_ub
    }
    trapz_kwargs = {
        't': t, 'dt': dt, 'axis': axis
    }

    ind_closed = np.array(
        is_closed(y, **ind_kwargs), dtype=np.float
    )
    closed_duration = np.trapz(ind_closed, **trapz_kwargs)
    duration = t[-1]-t[0]
    return closed_duration/duration

@_add_measure_docstring
def open_ratio(y, t=None, dt=1.0, axis=-1, closed_ub=0):
    """
    Return the open ratio

    This is the ratio of time spent open to total time

    """
    return 1 - closed_ratio(y, t=t, dt=dt, axis=axis, closed_ub=closed_ub)

@_add_measure_docstring
def closing_ratio(y, t=None, dt=1.0, axis=-1, closed_ub=0):
    """
    Return the closing ratio

    This is the ratio of time spent closing to the total time

    """
    # Create kwargs for the 'indicator' and 'numpy trapz' functions
    ind_kwargs = {
        't': t, 'dt': dt, 'closed_ub': closed_ub
    }
    trapz_kwargs = {
        't': t, 'dt': dt, 'axis': axis
    }

    ind_closing = np.array(is_closing(y, **ind_kwargs), dtype=np.float)
    closing_duration = np.trapz(ind_closing, **trapz_kwargs)
    duration = t[-1]-t[0]
    return closing_duration/duration

@_add_measure_docstring
def opening_ratio(y, t=None, dt=1.0, axis=-1, closed_ub=0):
    """
    Return the opening ratio

    This is the ratio of time spent opening to the total time

    """
    ind_kwargs = {
        't': t, 'dt': dt, 'closed_ub': closed_ub
    }
    trapz_kwargs = {
        't': t, 'dt': dt, 'axis': axis
    }

    ind_opening = np.array(
        is_opening(y, ind_kwargs), dtype=np.float)
    opening_duration = np.trapz(ind_opening, trapz_kwargs)
    duration = t[-1]-t[0]
    return opening_duration/duration

@_add_measure_docstring
def speed_ratio(y, t=None, dt=None, axis=-1, closed_ub=0):
    """
    Return the speed ratio

    This is the ratio of opening to closing times

    """
    kwargs = {
        't': t, 'dt': dt, 'axis': axis, 'closed_ub': closed_ub
    }
    return opening_ratio(y, **kwargs) / closing_ratio(y, **kwargs)

@_add_measure_docstring
def mfdr(y, t=None, dt=None, axis=-1, closed_ub=0):
    """
    Return the maximum flow declination rate (MFDR)

    """
    ind_kwargs = {
        't': t, 'dt': dt, 'closed_ub': closed_ub
    }

    if t is not None:
        _dt = t
    elif dt is not None:
        _dt = dt
    else:
        _dt = 1.0

    ind_open = is_open(y, **ind_kwargs)
    yp = np.gradient(y, _dt, axis=axis)
    return np.min(yp[ind_open])

@_add_measure_docstring
def ac_flow(y, t=None, dt=None, axis=-1, closed_ub=0):
    """
    Return the AC flow

    This is the amplitude from minimum to maximum of the signal

    """
    return np.max(y, axis=axis) - np.min(y, axis=axis)

@_add_measure_docstring
def acdc(y, t=None, dt=None, axis=-1, closed_ub=0):
    """
    See Holmberg et al. for the definition

    """
    y_ac = y - y.min()

    T = t[-1]-t[0]
    y_ac_rms = np.sqrt(np.trapz(t, y_ac**2)/T)
    y_ac_mean = np.trapz(t, y_ac)/T
    return y_ac_rms/y_ac_mean

@_add_measure_docstring
def rms_time(y, t=None, dt=None, axis=-1):
    """
    Return the RMS of a time-domain signal
    """
    return np.sqrt(np.mean(y**2, axis=axis))

## Frequency domain processing functions
# Signals
def prad_piston(y, f=None, df=1.0, axis=-1, piston_params=None):
    # handle depacking of the piston acoustic parameters
    if piston_params is None:
        piston_params = {
            'r': 1.0, 'theta': 0.0, 'a': 1.0, 'rho': 1.0, 'c': 343*100
        }

    r = piston_params['r']
    theta = piston_params['theta']
    a = piston_params['a']
    rho = piston_params['rho']
    c = piston_params['c']

    # compute the frequencies if not provided
    # this assumes `y` is a one-sided fourier representation and `y[0]`
    # corresponds to a frequency of 0.0 rad/s
    if f is None:
        f_shape = [1]*y.ndim
        f_shape[axis] = y.shape[axis]
        f = np.zeros(f_shape)
        f[:] = df*np.arange(y.shape[axis])

    # apply the piston-in-infinite-baffle formula to determine radiated pressure
    k = f/c
    zc = rho*c

    y = k*a*np.sin(theta)
    if theta == 0:
        return 1j/2*zc/r * y/np.pi * k * np.exp(-1j*k*r)
    else:
        return 1j/2*zc/r * y/np.pi * k * 2*sp.special.jv(1, y)/y * np.exp(-1j*k*r)


# Measures
def rms_freq(y, f=None, df=None, axis=-1):
    return np.sqrt(np.mean(y**2, axis=axis))

# def spl(y, t=None, dt=None, axis=-1):