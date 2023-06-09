"""
Test the module `fftutils`
"""

import pytest

import numpy as np
import matplotlib.pyplot as plt

from vfsig import fftutils

@pytest.fixture()
def n_time():
    """
    Return the number of time points
    """
    return 2**12

NUM_PERIOD = 10

@pytest.fixture()
def time_period(n_time):
    return np.linspace(0, NUM_PERIOD, n_time+1)[:-1]

@pytest.fixture()
def time(n_time):
    # The frequency is [cycles/s]
    FREQ = 100
    # The `(1/FREQ * NUM_PERIOD)` gives the total time of the signal
    # since it's [time/cycle] * [# of cycles]
    # Dividing by `n_time`, the number of time points in the signal, then gives
    # the time interval between points
    dt = (1/FREQ*NUM_PERIOD)/n_time
    return  dt*np.arange(n_time)

@pytest.fixture()
def ut(time_period):
    # The frequency of these signals is measures as cycles/[period]
    # i.e. a frequency of 1, will just produce a signal with the default frequency
    x = time_period
    return 5*np.sin(2*np.pi*x)

@pytest.fixture()
def vt(time_period):
    x = time_period
    return 5*np.sin(2*np.pi*x)

@pytest.fixture()
def ufft(ut):
    return np.fft.fft(ut)

@pytest.fixture()
def vfft(vt):
    return np.fft.fft(vt)

@pytest.fixture()
def urfft(ut):
    return np.fft.rfft(ut)

@pytest.fixture()
def vrfft(vt):
    return np.fft.rfft(vt)

@pytest.fixture()
def axis():
    return -1

def test_power_from_fft(ut, vt, ufft, vfft, axis):
    print(np.sum(np.conj(ut)*vt), fftutils.power_from_fft(ufft, vfft, axis=axis))

    assert np.isclose(
        np.sum(np.conj(ut)*vt, axis=axis),
        fftutils.power_from_fft(ufft, vfft, axis=axis)
    )

def test_power_from_rfft(ut, vt, urfft, vrfft, axis):
    assert np.isclose(
        np.sum(np.conj(ut)*vt, axis=axis),
        fftutils.power_from_rfft(urfft, vrfft, axis=axis, n=ut.shape[axis])
    )

