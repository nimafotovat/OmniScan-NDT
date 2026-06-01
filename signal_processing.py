import numpy as np
from scipy.signal import butter, filtfilt

def add_gaussian_noise(signal: np.ndarray, snr_db: float = 30.0) -> np.ndarray:
    """
    اضافه کردن نویز سفید گاوسی به سیگنال بر اساس نسبت سیگنال به نویز (SNR)
    """
   
    signal_power = np.mean(signal ** 2)
    
   
    if signal_power == 0:
        return signal + np.random.normal(0, 0.01, len(signal))
        
  
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return signal + noise

def apply_bandpass_filter(signal: np.ndarray, fs: float, center_freq: float, bandwidth_percent: float = 0.5, order: int = 5) -> np.ndarray:
    """
    اعمال فیلتر میان‌گذر باترورث متناسب با پهنای باند ترنسدیوسر
    استفاده از filtfilt برای حفظ فاز سیگنال و جلوگیری از تغییر زمان TOF (t=2d/v)
    """
    nyquist = 0.5 * fs
    
    
    lowcut = center_freq * (1 - bandwidth_percent)
    highcut = center_freq * (1 + bandwidth_percent)
    
   
    lowcut = max(0.01 * nyquist, lowcut)
    highcut = min(0.99 * nyquist, highcut)
    
    low = lowcut / nyquist
    high = highcut / nyquist
    
   
    b, a = butter(order, [low, high], btype='band')
    
  
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal
