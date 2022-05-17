import numpy as np

from vfsig import clinical

def setup_tdomain_signal():
    dt = 0.01
    t = dt*np.arange(1024)

    period = 2**4 * dt
    y = np.sin(2*np.pi * 1/period * t)
    return y, t, dt

def setup_fdomain_signal():
    y, t, dt = setup_tdomain_signal()

    fy = np.fft.rfft(y, y.size)
    freq = np.fft.rfftfreq(y.size, d=dt)
    dfreq = 1/(y.size * dt)

    fy_ = np.fft.fft(y, y.size)
    breakpoint()
    return fy, freq, dfreq

def test_is_closed(y, t, dt):
    clinical.is_closed(y, t, dt)

def test_prad_piston(fy, freq):
    return clinical.prad_piston(fy, freq)

if __name__ == '__main__':
    y, t, dt = setup_tdomain_signal()

    test_is_closed(y, t, dt)

    fy, freq, dfreq = setup_fdomain_signal()
    breakpoint()

    test_prad_piston(fy, freq)