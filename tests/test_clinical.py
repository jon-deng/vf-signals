import numpy as np

from vfsig import clinical

def setup_signal():
    dt = 0.01
    t = dt*np.arange(1024)

    period = 2**4 * dt
    y = np.sin(2*np.pi * 1/period * t)
    return y, t, dt

def test_is_closed(y, t, dt):
    clinical.is_closed(y, t, dt)

if __name__ == '__main__':
    y, t, dt = setup_signal()

    test_is_closed(y, t, dt)