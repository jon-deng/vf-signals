import numpy as np
import matplotlib.pyplot as plt

from vfsig import modal, clinical

def setup_1d_signal():
    N_PERIOD = 10.1
    N_PER_PERIOD = 128
    N_SAMPLE = int(round(N_PERIOD * N_PER_PERIOD))

    DT = 0.01

    t = DT*np.arange(N_SAMPLE)
    n_per_period = N_SAMPLE/N_PERIOD
    period = DT * n_per_period
    y = np.sin(2*np.pi * 1/period * t)
    return y, DT, int(round(N_PERIOD))

def setup_2d_signal_a():
    y, dt, nperiod = setup_1d_signal()

    # Return a (5, #samples) signal
    return y*np.ones((5, 1)), dt, nperiod

def setup_2d_signal_b():
    y, dt, nperiod = setup_1d_signal()

    # Return a (#samples, 5) signal
    return y[:, np.newaxis]*np.ones((1, 5)), dt, nperiod


def test_estimate_periodic_statistics(y, n_period):
    mean_y, stdev_y = modal.estimate_periodic_statistics(y, n_period)

    fig, ax = plt.subplots(1, 1)
    ax.plot(mean_y, color='b', ls='-')
    ax.plot(mean_y+stdev_y*3, color='b', ls='-.')
    ax.plot(mean_y-stdev_y*3, color='b', ls='-.')
    ax.set_xlabel("Time []")
    ax.set_ylabel("Signal []")
    fig.savefig('fig/test_estimate_periodic_statistics.png')

def test_estimate_fundamental_mode():
    y, dt, _ = setup_1d_signal()
    fund_f, fund_phase, df = modal.estimate_fundamental_mode(y, dt, axis=-1)
    print(fund_f.shape)

    y, dt, _ = setup_2d_signal_a()
    fund_f, fund_phase, df = modal.estimate_fundamental_mode(y, dt, axis=-1)
    print(fund_f.shape)

    y, dt, _ = setup_2d_signal_b()
    fund_f, fund_phase, df = modal.estimate_fundamental_mode(y, dt, axis=-2)
    print(fund_f.shape)

if __name__ == '__main__':
    y, dt, n_period = setup_1d_signal()

    fig, ax = plt.subplots(1, 1)
    t = dt*np.arange(y.size)
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

