import numpy as np
import matplotlib.pyplot as plt

from wave_math import generate_gaussian_pulse

fs = 100e6
t = np.arange(0, 5e-6, 1 / fs)
pulse = generate_gaussian_pulse(t, 5e6, 0.5, 2e-6)

plt.figure(figsize=(9, 3.5))
plt.plot(t * 1e6, pulse, color="#2E5A88")
plt.xlabel("Time (µs)")
plt.ylabel("Amplitude")
plt.title("Gaussian-modulated tone burst (5 MHz)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
