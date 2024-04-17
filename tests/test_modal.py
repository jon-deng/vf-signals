"""
Test the module `modal`
"""

import pytest

import numpy as np
import matplotlib.pyplot as plt

from vfsig import modal


@pytest.fixture()
def fundamental_period():
    return 1.0


@pytest.fixture()
def fundamental_freq(fundamental_period):
    return 1 / fundamental_period


@pytest.fixture()
def fundamental_phase():
    return np.random.rand() * 2 * np.pi


@pytest.fixture()
def time(fundamental_period):
    num_samples_per_period = 10
    num_periods = 20

    dt = fundamental_period / num_samples_per_period
    return dt * np.arange(num_samples_per_period * num_periods)


@pytest.fixture()
def signal(time, fundamental_period, fundamental_phase):
    return np.sin(2 * np.pi * time / fundamental_period - fundamental_phase)


def test_fundamental_mode_from_rfft(signal, time, fundamental_freq, fundamental_phase):
    est_fund_freq, dfreq, est_fund_phase, dphase, info = (
        modal.fundamental_mode_from_rfft(
            signal,
            time[1] - time[0],
        )
    )
    print(
        f"Estimated FO: {est_fund_freq:.2e} +/- {dfreq:.2e}, actual FO {fundamental_freq:.2e}"
    )
    print(
        f"Estimated phiO: {est_fund_phase:.2e} +/- {dphase:.2e}, actual phiO {fundamental_phase:.2e}"
    )


def test_fundamental_mode_from_peaks(signal, time, fundamental_freq, fundamental_phase):
    est_fund_freq, dfreq, est_fund_phase, dphase, info = (
        modal.fundamental_mode_from_peaks(
            signal,
            time[1] - time[0],
        )
    )
    print(
        f"Estimated FO: {est_fund_freq:.2e} +/- {dfreq:.2e}, actual FO {fundamental_freq:.2e}"
    )
    print(
        f"Estimated phiO: {est_fund_phase:.2e} +/- {dphase:.2e}, actual phiO {fundamental_phase:.2e}"
    )


# def test_estimate_periodic_statistics(y, n_period):
#     mean_y, stdev_y = modal.periodic_stats(y, n_period)

#     fig, ax = plt.subplots(1, 1)
#     ax.plot(mean_y, color='b', ls='-')
#     ax.plot(mean_y+stdev_y*3, color='b', ls='-.')
#     ax.plot(mean_y-stdev_y*3, color='b', ls='-.')
#     ax.set_xlabel("Time []")
#     ax.set_ylabel("Signal []")
#     fig.savefig('fig/test_estimate_periodic_statistics.png')


if __name__ == '__main__':
    y, dt, n_period = setup_1d_signal()

    fig, ax = plt.subplots(1, 1)
    t = dt * np.arange(y.size)
    ax.plot(t, y)
    ax.set_xlabel("Time []")
    ax.set_ylabel("Signal []")
    fig.savefig('fig/test_modal.png')

    # Note that the signal generated in `setup_signal` has no uncertainty;
    # there will be a non-zero uncertainty mainly due to the fact that FFT
    # always guesses an integer number of periods in the signal.
    # If the signal contains 4.5 periods, FFT will guess that the signal has
    # either 4 or 5 periods which means that succesive periods will not overlap
    # perfectly
    # The frequency mismatch will cause some kind of aliasing uncertainty in the
    # periodic signal
    # You can play with this in `setup_signal` by setting `N_PERIOD` to non-integer periods
    # Intuitively, the worst case might happen when there are N+0.5 periods as the extra half
    # period will misalign things
    test_estimate_periodic_statistics(y, n_period)

    test_estimate_fundamental_mode()
