import numpy as np
import matplotlib.pyplot as plt

from vfsig import fftutils

def setup_signals():
    """
    Setup two u and v signals in the time domains
    """
    N = 2**12
    NUM_PERIOD = 10

    x = np.linspace(0, NUM_PERIOD, N+1)[:-1]
    u = 5*np.sin(2*np.pi*x)

    # Shift v forward by 1/10 cycle relative to u
    x = np.linspace(0, NUM_PERIOD, N+1)[:-1] - 0.1
    v = np.sin(2*np.pi*x)

    FREQ = 100
    dt = (1/FREQ*NUM_PERIOD)/N
    t = dt*np.arange(N)

    return t, u, v

if __name__ == '__main__':
    t, u, v = setup_signals()
    print(u.size, v.size)

    fig, ax = plt.subplots(1, 1)
    ax.plot(t, u, label="u")
    ax.plot(t, v, label="v")
    fig.savefig("test_fftutils.png")

    fft_u = np.fft.fft(u)
    fft_v = np.fft.fft(v)

    print(np.sum(np.conj(u)*v), fftutils.power_from_fft(fft_u, fft_v))

    rfft_u = np.fft.rfft(u)
    rfft_v = np.fft.rfft(v)
    breakpoint()
    print(np.sum(np.conj(u)*v), fftutils.power_from_rfft(rfft_u, rfft_v, n=u.size))
