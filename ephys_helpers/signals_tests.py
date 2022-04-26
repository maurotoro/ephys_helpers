import numpy as np


freqs_n = 50

x = np.arange(0, 5, step=1 / 3e4)

freqs_vals = np.logspace(1, 4, num=freqs_n).astype(int)
amps_vals = np.logspace(4, 1, num=freqs_n)/1000
phase_shifts = np.random.uniform(0, 2 * np.pi, size=freqs_n)

ys = [
    amp * np.sin(2 * np.pi * freq * x + phase)
    for amp, freq, phase in zip(amps_vals, freqs_vals, phase_shifts)
]

y = np.sum(ys, axis=0)
