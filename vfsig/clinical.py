"""
Functions for computing common clinical measures of speech signals

The `..._ratio` function definitions come from the paper (Holmberg et al., 1998)

Citations
---------
E. Holmberg, R. Hillman, and J. Perkell -- Glottal airflow and transglottal air pressure measurements for male and female speakers in soft, normal, and loud voice -- 1998 -- JASA
"""

from typing import Optional, Mapping, Union, Callable, Tuple, Any
# from numpy.typing import
import numpy as np
import scipy as sp

RealArray = np.ndarray[float]
ComplexArray = np.ndarray[complex]
BoolArray = np.ndarray[bool]


#### Time domain clinical measures

# All time domain clinical measures have the basic function signature

# Basic time domain functions have the signature:
# Kwargs = Mapping[str, Any]
TimeDomainFunction = Callable[[RealArray, RealArray], RealArray]
# The basic signature is augmented to
# `def g(y: RealArray, t: Union[RealArray, float], axis: Optional[int], **kwargs)`
# by the decorator `_add_optional_kwargs`

# This adds augmented signature parameters to the docstring
def _add_standard_parameters_to_docstring(time_domain_function: TimeDomainFunction):
    """
    Add the 'Parameters' sections to a time domain function's docstring
    """

    doc_standard_parameters = """y : RealArray of shape (..., N)
        The time domain signal
    t : Optional[RealArray] of shape (..., N)
        Sample times

        This should have a shape broadcastable to `y`
    dt : Optional[float]
        An optional uniform sampling time

        If `t` is supplied  `dt` will be ignored.
        If `dt` is supplied and `t=None` sample times are assumed to be evenly spaced at
        `dt`.
    axis: Optional[int]
        The time axis for both `y` and `t`"""
    time_domain_function.__doc__ = time_domain_function.__doc__.format(
        doc_standard_parameters
    )
    return time_domain_function

## State indicator functions
# These time domain functions return a boolean array indicating whether the VFs
# are/aren't in some state (open, closed, closing, opening)
# The basic signature is
# `def f(y: RealArray, t: RealArray, closed_ub: Optional[float]=0)`

# Decorator for adding `axis` and optional `time` and `dt` kwargs
def _add_optional_kwargs(time_domain_function: TimeDomainFunction):
    def dec_func(
        y: RealArray,
        t: RealArray = None,
        dt: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        **kwargs
    ) -> Union[RealArray, ComplexArray, BoolArray]:
        if t is None:
            time = dt * np.arange(y.shape[-1])
        else:
            time = t

        y = np.moveaxis(y, axis, -1)
        time = np.moveaxis(time, axis, -1)
        # NOTE: This moves the time axis from -1 back to its original spot
        # You need this because these functions don't reduce the time axis
        return np.moveaxis(time_domain_function(y, time, **kwargs), axis, -1)

    dec_func.__doc__ = time_domain_function.__doc__

    return dec_func

@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def is_closed(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> BoolArray:
    """
    Return a boolean array indicating if VFs are closed

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    BoolArray
        Indicator if glottis is closed
    """
    return y < closed_ub

@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def is_open(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> BoolArray:
    """
    Return a boolean array indicating if VFs are opening

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    BoolArray
        Indicator if glottis is open
    """
    return np.logical_not(is_closed(y, t=t, closed_ub=closed_ub))


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def is_closing(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> BoolArray:
    """
    Return a boolean array indicating if VFs are closing

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    BoolArray
        Indicator if glottis is closing
    """
    axis = -1
    y_prime = np.gradient(y, t, axis=axis)
    return np.logical_and(is_open(y, t=t, closed_ub=closed_ub), y_prime < 0)


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def is_opening(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> BoolArray:
    """
    Return a boolean array indicating if VFs are opening

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    BoolArray
        Indicator if glottis is opening
    """
    axis = -1
    y_prime = np.gradient(y, t, axis=axis)
    return np.logical_and(is_open(y, t=t, closed_ub=closed_ub), y_prime >= 0)


## Scalar signal summaries
# These time domain functions reduce a time signal into a scalar measure

# Decorator for adding `axis` and optional `time` and `dt` kwargs
def _add_optional_kwargs(time_domain_function):
    def dec_func(
        y: RealArray,
        t: RealArray = None,
        dt: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        **kwargs
    ) -> Union[RealArray, ComplexArray, BoolArray]:
        if t is None:
            time = dt * np.arange(y.shape[-1])
        else:
            time = t

        y = np.moveaxis(y, axis, -1)
        time = np.moveaxis(time, axis, -1)
        # NOTE: You don't need `moveaxis` since the time axis is reduced
        return time_domain_function(y, time, **kwargs)

    dec_func.__doc__ = time_domain_function.__doc__

    return dec_func

def _duration(t: RealArray) -> float:
    return t[..., -1] - t[..., 0]


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def closed_ratio(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the closed ratio

    This is the ratio of time spent closed to total time

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The closed ratio
    """
    ind_closed = np.array(is_closed(y, t, closed_ub=closed_ub), dtype=float)
    closed_duration = np.trapz(ind_closed, x=t, axis=-1)
    duration = _duration(t)
    return closed_duration / duration


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def open_ratio(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the open ratio

    This is the ratio of time spent open to total time

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The open ratio
    """
    return 1 - closed_ratio(y, t, closed_ub=closed_ub)


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def closing_ratio(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the closing ratio

    This is the ratio of time spent closing to the total time

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The closing ratio
    """
    ind_closing = np.array(is_closing(y, t, closed_ub), dtype=float)
    closing_duration = np.trapz(ind_closing, x=t, axis=-1)
    duration = _duration(t)
    return closing_duration / duration


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def opening_ratio(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the opening ratio

    This is the ratio of time spent opening to the total time

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The opening ratio
    """
    ind_opening = np.array(is_opening(y, t, closed_ub), dtype=float)
    opening_duration = np.trapz(ind_opening, x=t, axis=-1)
    duration = _duration(t)
    return opening_duration / duration


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def speed_ratio(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the speed ratio

    This is the ratio of opening to closing times

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The speed ratio
    """
    return opening_ratio(y, t, closed_ub=closed_ub) / closing_ratio(
        y, t, closed_ub=closed_ub
    )


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def mfdr(y: RealArray, t: RealArray, closed_ub: Optional[float] = 0) -> RealArray:
    """
    Return the maximum flow declination rate (MFDR)

    Parameters
    ----------
    {}
    closed_ub : Optional[float]
        The upper bound of `y` where the glottis is considered closed

        Closure is considered to happen when `y < closed_ub`

    Returns
    -------
    RealArray
        The MFDR
    """
    ind_open = is_open(y, t, closed_ub)
    yp = np.gradient(y, t, axis=-1)
    return np.min(yp[ind_open])


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def ac_flow(y: RealArray, t: RealArray) -> RealArray:
    """
    Return the AC flow

    This is the amplitude from minimum to maximum of the signal

    Parameters
    ----------
    {}

    Returns
    -------
    RealArray
        The AC flow
    """
    return np.max(y, axis=-1) - np.min(y, axis=-1)


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def acdc(y: RealArray, t: RealArray) -> RealArray:
    """
    See Holmberg et al. for the definition

    Parameters
    ----------
    {}

    Returns
    -------
    RealArray
        The acdc ratio
    """
    y_ac = y - y.min(axis=-1, keepdims=True)

    T = _duration(t)
    y_ac_rms = np.sqrt(np.trapz(t, y_ac**2, axis=-1) / T)
    y_ac_mean = np.trapz(t, y_ac, axis=-1) / T
    return y_ac_rms / y_ac_mean


@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def rms_time(y: RealArray, t: RealArray) -> RealArray:
    """
    Return the RMS of a time-domain signal

    Parameters
    ----------
    {}

    Returns
    -------
    RealArray
        The signal RMS
    """
    return np.sqrt(np.mean(y**2, axis=-1))


#### Frequency domain clinical measures

# All frequency domain clinical measures have the basic function signature
FreqDomainFunction = Callable[[ComplexArray, RealArray], ComplexArray]
#
# The basic signature is augmented to
# `def g(y: RealArray, f: Optional[RealArray], df: Optional[float] axis: Optional[int], **kwargs)`
# by the decorator `_add_optional_kwargs`
def _add_standard_parameters_to_docstring(freq_domain_function: FreqDomainFunction):
    """
    Add the 'Parameters' sections to a frequency domain function's docstring
    """

    doc_standard_parameters = """y : ComplexArray of shape (..., N)
        The time domain signal
    f : Optional[RealArray] of shape (..., N)
        Frequency bins

        This should have a shape broadcastable to `y`
    df : Optional[float]
        An optional frequency spacing

        If `f` is supplied  `df` will be ignored.
        If `df` is supplied and `f=None` the frequency bins are assumed to increase from
        0 in increments of `df`.
    axis: Optional[int]
        The frequency axis for both `y` and `f`"""
    freq_domain_function.__doc__ = freq_domain_function.__doc__.format(
        doc_standard_parameters
    )
    return freq_domain_function

# Decorator for adding `df` and `axis` kwargs
def _add_optional_kwargs(freq_domain_function: FreqDomainFunction):
    def dec_func(
        y: ComplexArray,
        f: Optional[RealArray] = None,
        df: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        **kwargs
    ) -> ComplexArray:
        if f is None:
            f = df * np.arange(y.shape[-1])
        else:
            f = f

        y = np.moveaxis(y, axis, -1)
        f = np.moveaxis(f, axis, -1)
        return np.moveaxis(freq_domain_function(y, f, **kwargs), -1, axis)

    dec_func.__doc__ = freq_domain_function.__doc__

    return dec_func

@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def prad_piston(
    q: ComplexArray,
    f: Optional[RealArray] = None,
    piston_params: Optional[Mapping[str, float]] = None,
) -> ComplexArray:
    """
    Return the complex pressure amplitude from a flow source in the frequency domain

    The complex pressure amplitude assumes the flow source acts as a piston in
    and infinite baffle. The resulting complex pressure given the complex flow
    is given by Kinsler et al. "Fundamental of Acoustics"
    [Section 7.4, equation 7.4.17].

    Parameters
    ----------
    {}
    piston_params: Mapping[str, float]
        A mapping of named piston parameters to values. The parameters are given
        by (see Figure 7.4.3 of Kinsler):
        'a' - radius of the piston
        'r' - distance from the piston center
        'theta' - angle from the piston central axis
        'rho' - density of raidating material (usu. air)
        'c' - speed of sound in material

    Returns
    -------
    prad: ComplexArray of shape (..., N)
        The complex radiated pressure in the frequency domain.
    """
    # Handle depacking of the piston-in-baffle approximation's acoustic parameters
    default_piston_params = {
        'r': 100.0,
        'theta': 0.0,
        'a': 1.0,
        'rho': 0.001225,
        'c': 343 * 100,
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

    # apply the piston-in-infinite-baffle formula to determine radiated pressure
    # k is the wave number
    k = f / c
    zc = rho * c

    # NOTE: the below formula are missing factors of 'a' relative to eq. (7.4.17)
    # because this formula uses flow rate, instead of piston velocity.
    if theta == 0:
        return 1j / 2 * zc * q / np.pi * 1 / r * k * np.exp(-1j * k * r)
    else:
        y = k * a * np.sin(theta)
        return (
            1j
            / 2
            * zc
            * q
            / np.pi
            * 1
            / r
            * k
            * 2
            * sp.special.jv(1, y)
            / y
            * np.exp(-1j * k * r)
        )

# Decorator for adding `df` and `axis` kwargs
def _add_optional_kwargs(freq_domain_function: FreqDomainFunction):
    def dec_func(
        y: ComplexArray,
        f: Optional[RealArray] = None,
        df: Optional[float] = 1.0,
        axis: Optional[int] = -1,
        **kwargs
    ) -> ComplexArray:
        if f is None:
            f = df * np.arange(y.shape[-1])
        else:
            f = f

        y = np.moveaxis(y, axis, -1)
        f = np.moveaxis(f, axis, -1)
        # NOTE: You don't have to move the axis since the frequency axis is reduced
        return freq_domain_function(y, f, **kwargs)

    dec_func.__doc__ = freq_domain_function.__doc__

    return dec_func

@_add_standard_parameters_to_docstring
@_add_optional_kwargs
def rms_freq(y: ComplexArray, f: RealArray) -> ComplexArray:
    """
    Return the RMS of a frequency domain signal

    Parameters
    ----------
    {}

    Returns
    -------
    ComplexArray of shape (..., N)
        The complex RMS
    """
    return np.sqrt(np.mean(y**2, axis=-1))
