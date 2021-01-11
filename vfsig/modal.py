import numpy as np
from numpy import fft

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

