"""
This module contains functionality for calculating common clinical measures of speech signals

The `..._ratio` function definitions come from the paper (Holmberg et al., 1998)

Citations
---------
E. Holmberg, R. Hillman, and J. Perkell -- Glottal airflow and transglottal air pressure measurements for male and female speakers in soft, normal, and loud voice -- 1998 -- JASA
"""
from typing import Optional
from numpy.typing import NDArray
import numpy as np
import scipy as sp

## State indicator functions
# These functions indicate whether the VFs are in some state (open, closed,
# etc..) or not

# This adds the 'Parameters' and 'Returns' docstring sections that are common
# to all the state indicator functions
def _add_state_indicator_docstring(signal_function):
    add_docstring = """

    Parameters
    ----------
    y : NDArray of shape (..., N)
        A signal of glottal width or glottal flow
    t : Optional[NDArray] of shape (..., N)
        Time instances of the glottal signal. This should have a shape
        broadcastable to `y`
    dt : Optional[float]
        The uniform time spacing between signal samples. If `t` is
        supplied, values of `dt` will be ignored
    closed_ub : Optional[float]
        The value of `y` where for all `y < closed_ub`, the signal is assumed to
        indicate closure.

    Returns
    -------
    array_like
        Boolean array indicating whether the state is satisfied or not
    """
    signal_function.__doc__ = signal_function.__doc__ + add_docstring
    return signal_function

def _add_optional_time(state_indicator_function):
    def new_state_indicator_function(
            y: SignalArray, t: TimeArray=None, dt: float=1.0,
            closed_ub: Optional[float]=0.0
        ):
        if t is None:
            time = dt*np.arange(y.shape[-1])
        else:
            time = t
        return state_indicator_function(y, time, closed_ub=closed_ub)

    return new_state_indicator_function

TimeArray = Optional[NDArray[bool]]
SignalArray = NDArray[float]

@_add_optional_time
@_add_state_indicator_docstring
def is_closed(y: SignalArray, t: TimeArray, closed_ub: Optional[float]=0) -> NDArray[bool]:
    """
    Return a boolean array indicating if VFs are closed
    """
    return y < closed_ub

@_add_optional_time
@_add_state_indicator_docstring
def is_open(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return a boolean array indicating if VFs are opening
    """
    return np.logical_not(
        is_closed(y, t=t, closed_ub=closed_ub)
    )

@_add_optional_time
@_add_state_indicator_docstring
def is_closing(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return a boolean array indicating if VFs are closing
    """
    axis = -1
    y_prime = np.gradient(y, t, axis=axis)
    return np.logical_and(
        is_open(y, t=t, closed_ub=closed_ub),
        y_prime < 0
    )

@_add_optional_time
@_add_state_indicator_docstring
def is_opening(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return a boolean array indicating if VFs are opening
    """
    axis = -1
    y_prime = np.gradient(y, t, axis=axis)
    return np.logical_and(
        is_open(y, t=t, closed_ub=closed_ub),
        y_prime >= 0
    )

## Return scalar summaries of the signal
def _duration(t: TimeArray, axis: Optional[float]=-1):
    return t[-1]-t[0]

def _add_measure_docstring(measure_function):
    add_docstring = """

    Parameters
    ----------
    y : NDArray of shape (..., N)
        A signal of glottal width or glottal flow
    t : Optional[NDArray] of shape (..., N)
        Time instances of the glottal signal. This should have a shape
        broadcastable to `y`
    dt : Optional[float]
        The uniform time spacing between signal samples. If `t` is
        supplied, values of `dt` will be ignored
    closed_ub : Optional[float]
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
def closed_ratio(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the closed ratio

    This is the ratio of time spent closed to total time
    """
    axis = -1
    trapz_kwargs = {'x': t, 'axis': axis}

    ind_closed = np.array(is_closed(y, t, closed_ub), dtype=float)
    closed_duration = np.trapz(ind_closed, **trapz_kwargs)
    duration = _duration(t, axis)
    return closed_duration/duration

@_add_measure_docstring
def open_ratio(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the open ratio

    This is the ratio of time spent open to total time
    """
    return 1 - closed_ratio(y, t , closed_ub)

@_add_measure_docstring
def closing_ratio(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the closing ratio

    This is the ratio of time spent closing to the total time
    """
    # Create kwargs for the 'indicator' and 'numpy trapz' functions
    axis = -1
    trapz_kwargs = {'x': t, 'axis': axis}

    ind_closing = np.array(is_closing(y, t, closed_ub), dtype=float)
    closing_duration = np.trapz(ind_closing, **trapz_kwargs)
    duration = _duration(t, axis)
    return closing_duration/duration

@_add_measure_docstring
def opening_ratio(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the opening ratio

    This is the ratio of time spent opening to the total time
    """
    axis = -1
    trapz_kwargs = {'x': t, 'axis': axis}

    ind_opening = np.array(is_opening(y, t, closed_ub), dtype=float)
    opening_duration = np.trapz(ind_opening, **trapz_kwargs)
    duration = _duration(t, axis)
    return opening_duration/duration

@_add_measure_docstring
def speed_ratio(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the speed ratio

    This is the ratio of opening to closing times
    """
    return opening_ratio(y, t, closed_ub) / closing_ratio(y, t, closed_ub)

@_add_measure_docstring
def mfdr(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the maximum flow declination rate (MFDR)
    """
    axis = -1
    ind_open = is_open(y, t, closed_ub)
    yp = np.gradient(y, t, axis=axis)
    return np.min(yp[ind_open])

@_add_measure_docstring
def ac_flow(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the AC flow

    This is the amplitude from minimum to maximum of the signal
    """
    axis = -1
    return np.max(y, axis=axis) - np.min(y, axis=axis)

@_add_measure_docstring
def acdc(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    See Holmberg et al. for the definition
    """
    axis = -1
    y_ac = y - y.min(axis=-1, keepdims=True)

    T = _duration(t, axis)
    y_ac_rms = np.sqrt(np.trapz(t, y_ac**2, axis=axis)/T)
    y_ac_mean = np.trapz(t, y_ac, axis=axis)/T
    return y_ac_rms/y_ac_mean

@_add_measure_docstring
def rms_time(y: SignalArray, t: TimeArray, closed_ub=0):
    """
    Return the RMS of a time-domain signal
    """
    axis = -1
    return np.sqrt(np.mean(y**2, axis=axis))

## Frequency domain processing functions

# Signals
def prad_piston(q, f=None, df=1.0, axis=-1, piston_params=None):
    """
    Return the complex pressure amplitude from a flow source

    The complex pressure amplitude assumes the flow source acts as a piston in
    and infinite baffle. The resulting complex pressure given the complex flow
    is given by Kinsler et al. "Fundamental of Acoustics"
    [Section 7.4, equation 7.4.17].

    Parameters
    ----------
    q : np.array
        The frequency domain components of q
    f : np.array
        Frequencies corresponding to components of `q` in [rad/unit time]
    df : float
        Frequency spacing of `q` components in [rad/unit time]
    piston_params : dict
        A mapping of named piston parameters to values. The parameters are given
        by (see Figure 7.4.3 of Kinsler):
        'a' - radius of the piston
        'r' - distance from the piston center
        'theta' - angle from the piston central axis
        'rho' - density of raidating material (usu. air)
        'c' - speed of sound in material
    """
    # handle depacking of the piston acoustic parameters
    default_piston_params = {
        'r': 100.0, 'theta': 0.0, 'a': 1.0, 'rho': 0.001225, 'c': 343*100
    }
    if piston_params is None:
        piston_params = {}
    default_piston_params.update(piston_params)
    piston_params = default_piston_params

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
    # k is the wave number
    k = f/c
    zc = rho*c

    # Note the below formula are missing factors of 'a' relative to eq. (7.4.17)
    # because this formula uses flow rate, instead of piston velocity.
    if theta == 0:
        return 1j/2 * zc * q/np.pi * 1/r * k * np.exp(-1j*k*r)
    else:
        y = k*a*np.sin(theta)
        return 1j/2 * zc * q/np.pi * 1/r * k * 2*sp.special.jv(1, y)/y * np.exp(-1j*k*r)

# Measures
def rms_freq(y, f=None, df=None, axis=-1):
    return np.sqrt(np.mean(y**2, axis=axis))

# def spl(y, t=None, dt=None, axis=-1):
