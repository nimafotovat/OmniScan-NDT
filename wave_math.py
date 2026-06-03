import numpy as np


def generate_gaussian_pulse(time_array, center_freq, bandwidth_factor, time_delay, amplitude=1.0):
    sigma = 1.0 / (2 * np.pi * center_freq * bandwidth_factor)
    env = np.exp(-0.5 * ((time_array - time_delay) / sigma) ** 2)
    carrier = np.sin(2 * np.pi * center_freq * (time_array - time_delay))
    return amplitude * env * carrier
