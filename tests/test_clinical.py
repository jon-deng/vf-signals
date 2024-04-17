"""
Test the module `clinical`
"""

import pytest

import numpy as np

from vfsig import clinical


@pytest.fixture(params=[(5,), (5, 4), (6, 2, 100)])
def signal_shape(request):
    return request.param


@pytest.fixture()
def time(signal_shape):
    return np.linspace(0, 1, signal_shape[-1])


@pytest.fixture()
def time_intercept(time, signal_shape):
    tini, tfin = time[..., 0], time[..., -1]
    u = np.random.rand(*(signal_shape[:-1] + (1,)))
    return u / 1 * (tfin - tini) + tini


@pytest.fixture()
def y_slope():
    return 2 * (np.random.rand() - 0.5)


@pytest.fixture()
def y_linear(signal_shape, time, time_intercept, y_slope):
    y = np.zeros(signal_shape)
    y[:] = y_slope * (time - time_intercept)
    return y


# To test if signals are open/closed, generate a boolean array indicating
# whether the signal should be open or closed, then generate a signal
# according to that
@pytest.fixture()
def is_closed(signal_shape):
    return np.random.randint(0, 2, size=signal_shape, dtype=bool)


@pytest.fixture()
def y_closed(is_closed):
    y = np.zeros(is_closed.shape)
    y[is_closed] = -1.0
    y[np.logical_not(is_closed)] = 1.0
    return y


class TestStateIndicatorFunctions:

    def test_is_closed(self, y_closed, is_closed):
        assert np.all(clinical.is_closed(y_closed) == is_closed)

    def test_is_open(self, y_closed, is_closed):
        assert np.all(clinical.is_open(y_closed) != is_closed)

    # To test if closing/opening phases are detected properly, generate linearly
    # increasing/decreasing signals that cross the x-axis

    @pytest.fixture()
    def y_is_closing(self, y_linear, y_slope):
        """ """
        is_open = clinical.is_open(y_linear)
        if y_slope < 0:
            return is_open
        else:
            return np.zeros(y_linear.shape, dtype=bool)

    def test_is_closing(self, y_linear, time, y_is_closing):
        assert np.all(clinical.is_closing(y_linear, time, closed_ub=0) == y_is_closing)

    @pytest.fixture()
    def y_is_opening(self, y_linear, y_slope):
        """ """
        is_open = clinical.is_open(y_linear)
        if y_slope > 0:
            return is_open
        else:
            return np.zeros(y_linear.shape, dtype=bool)

    def test_is_opening(self, y_linear, time, y_is_opening):
        assert np.all(clinical.is_opening(y_linear, time, closed_ub=0) == y_is_opening)


class TestScalarMeasures:
    """
    Test the signal measures defined in (Holmberg et al., 1988)
    """

    # To test the closed ratio is correct, we use the fact that integrating an
    # indicator function for closure with `np.trapz` counts:
    #   +1 if VFs are closed at both endpoints of the time interval
    #   +0.5 if VFs are closed/open at endpoints of the time interval
    #   +0 if VFs are open at both endpoints of the time interval
    # TODO: This is probably not a great test but I can't think of what else to do
    @pytest.fixture()
    def closed_ratio(self, is_closed, time):
        is_transition_interval = is_closed[..., :-1] != is_closed[..., 1:]
        not_transition_interval = np.logical_not(is_transition_interval)
        is_closed_interval = np.logical_and(is_closed[..., :-1], is_closed[..., 1:])
        not_closed_interval = np.logical_not(is_closed_interval)

        # Multiply by the ones array to make sure `dt` has the right broadcasted shape
        dt = np.ones(is_closed_interval.shape) * (time[..., 1:] - time[..., :-1])
        dt[np.logical_and(not_closed_interval, not_transition_interval)] = 0
        dt[is_transition_interval] = 0.5 * dt[is_transition_interval]

        t_total = time[..., -1] - time[..., 0]
        t_closed = np.sum(dt, axis=-1)
        return t_closed / t_total

    @pytest.fixture()
    def y_closed(self, is_closed, time):
        y_is_open = np.logical_not(is_closed)
        y = np.ones(is_closed.shape)
        y[is_closed] = np.random.rand(is_closed.sum()) - 1
        y[y_is_open] = np.random.rand((y_is_open).sum())
        return y

    def test_closed_ratio(self, y_closed, time, closed_ratio):
        _closed_ratio = clinical.closed_ratio(y_closed, time)
        assert np.all(np.isclose(_closed_ratio, closed_ratio))


# def setup_tdomain_signal():
#     dt = 0.01
#     t = dt*np.arange(1024)

#     period = 2**4 * dt
#     y = np.sin(2*np.pi * 1/period * t)
#     return y, t, dt

# def setup_fdomain_signal():
#     y, t, dt = setup_tdomain_signal()

#     fy = np.fft.rfft(y, y.size)
#     freq = np.fft.rfftfreq(y.size, d=dt)
#     dfreq = 1/(y.size * dt)

#     fy_ = np.fft.fft(y, y.size)
#     breakpoint()
#     return fy, freq, dfreq

# def test_is_closed(y, t, dt):
#     clinical.is_closed(y, t, dt)

# def test_prad_piston(fy, freq):
#     return clinical.prad_piston(fy, freq)

# if __name__ == '__main__':
#     y, t, dt = setup_tdomain_signal()

#     test_is_closed(y, t, dt)

#     fy, freq, dfreq = setup_fdomain_signal()
#     breakpoint()

#     test_prad_piston(fy, freq)
