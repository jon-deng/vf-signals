"""
miscellaneous functions
"""

import numpy as np
import scipy as sp


def resample_over_uniform_period(t, y, tperiod, fs, axis=0, truncate_final_period=True):
    """
    Resample a signal with samples evenly distributed over a period
    """
    dt = t[-1] - t[0]
    nperiod = (
        int(np.floor(dt / tperiod))
        if truncate_final_period
        else int(np.ceil(dt / tperiod))
    )

    interpolate = sp.interpolate.interp1d(t, y, axis=axis)
    tr = np.linspace(0.0, nperiod * tperiod, fs * nperiod + 1)
    yr = interpolate(tr)
    return tr, yr
