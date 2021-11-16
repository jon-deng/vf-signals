import numpy as np
from numpy import fft

def estimate_fundamental_mode(t, y, flb=0, fub=np.inf):
    """
    Return information about the modes of a signal

    Returns
    -------
    f0 : float
        Fundamental frequency estimate
    phi0 : float
        Phase of fundamental mode
    """
    # Remove the mean component of y
    y_ = y - y.mean()

    # Compute the DFT and frequency components
    dfty = fft.rfft(y_)
    f = fft.rfftfreq(t.size, d=(t[-1]-t[0])/(t.size-1))

    idx_inrange = np.logical_and(f>flb, f<fub)
    f = f[idx_inrange]
    dfty = dfty[idx_inrange]

    idx_f0 = np.argmax(np.abs(dfty))

    return f[idx_f0], np.angle(dfty[idx_f0])

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

