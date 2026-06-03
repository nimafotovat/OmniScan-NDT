import numpy as np
from scipy.signal import butter, filtfilt, hilbert


def add_gaussian_noise(signal, snr_db=30.0):
    peak = np.max(np.abs(signal))
    if peak == 0:
        return signal + np.random.normal(0, 0.01, len(signal))
    noise_power = (peak ** 2) / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    return signal + noise


def compute_envelope(signal):
    return np.abs(hilbert(signal))


def apply_bandpass_filter(signal, fs, center_freq, bandwidth_percent=0.5, order=5):
    nyq = 0.5 * fs
    lowcut = max(0.01 * nyq, center_freq * (1 - bandwidth_percent))
    highcut = min(0.99 * nyq, center_freq * (1 + bandwidth_percent))
    if lowcut >= highcut:
        return signal.copy()
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, signal)
