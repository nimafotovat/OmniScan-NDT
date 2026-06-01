import numpy as np
from scipy.signal import butter, filtfilt

def add_gaussian_noise(signal: np.ndarray, snr_db: float) -> np.ndarray:
    """
    اضافه کردن نویز سفید گاوسی به سیگنال بر اساس نسبت سیگنال به نویز (SNR)
    """
    # محاسبه توان سیگنال
    signal_power = np.mean(signal ** 2)
    
    # محاسبه توان نویز بر اساس فرمول SNR
    noise_power = signal_power / (10 ** (snr_db / 10))
    
    # تولید نویز گاوسی
    noise = np.random.normal(0, np.sqrt(noise_power), len(signal))
    
    return signal + noise

def apply_bandpass_filter(signal: np.ndarray, lowcut: float, highcut: float, sample_rate: float, order: int = 5) -> np.ndarray:
    """
    اعمال فیلتر میان‌گذر باترورث برای شبیه‌سازی پهنای باند دستگاه و حذف نویزهای فرکانس بالا/پایین
    """
    nyquist = 0.5 * sample_rate
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # طراحی فیلتر باترورث
    b, a = butter(order, [low, high], btype='band')
    
    # استفاده از filtfilt برای جلوگیری از تغییر فاز سیگنال (Zero-phase filtering)
    filtered_signal = filtfilt(b, a, signal)
    
    return filtered_signal