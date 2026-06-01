import numpy as np

def generate_gaussian_pulse(time_array: np.ndarray, center_freq: float, bandwidth_factor: float, time_delay: float, amplitude: float = 1.0) -> np.ndarray:
    """
    تولید پالس اولیه ترنسدیوسر (سیگنال گوسی مدوله شده).
    """
    sigma = 1.0 / (2 * np.pi * center_freq * bandwidth_factor)
    envelope = np.exp(-0.5 * ((time_array - time_delay) / sigma) ** 2)
    carrier = np.sin(2 * np.pi * center_freq * (time_array - time_delay))
    
    pulse = amplitude * envelope * carrier
    return pulse