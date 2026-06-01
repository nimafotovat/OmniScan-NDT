import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_engine import UltrasonicSimulationEngine
from signal_processing import add_gaussian_noise, apply_bandpass_filter

# ۱. تنظیمات اولیه 
STEEL_VELOCITY = 5900.0      
STEEL_ATTENUATION = 1.5      
SAMPLE_RATE = 100e6

engine = UltrasonicSimulationEngine(sound_velocity=STEEL_VELOCITY, attenuation_coeff=STEEL_ATTENUATION, sample_rate=SAMPLE_RATE)


defect_depth_m = 0.02
probe_freq = 5e6
time_array, clean_echo = engine.simulate_echo(
    defect_depth=defect_depth_m, reflection_coeff=0.5, probe_freq=probe_freq, bw_factor=0.5
)


noisy_echo = add_gaussian_noise(clean_echo, snr_db=20.0)


filtered_echo = apply_bandpass_filter(
    signal=noisy_echo, lowcut=2e6, highcut=8e6, sample_rate=SAMPLE_RATE, order=4
)


depth_array_mm = (time_array * STEEL_VELOCITY / 2) * 1000


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)


ax1.plot(depth_array_mm, noisy_echo, color='gray', linewidth=1)
ax1.set_title('Raw A-Scan Signal with Gaussian Noise', fontsize=11, fontweight='bold')
ax1.set_ylabel('Amplitude', fontsize=10)
ax1.grid(True, linestyle='--', alpha=0.5)


ax2.plot(depth_array_mm, filtered_echo, color='#004d99', linewidth=1.5)
ax2.axvline(x=defect_depth_m * 1000, color='red', linestyle='--', linewidth=1, label='Actual Defect Location (20 mm)')
ax2.set_title('Filtered A-Scan Signal (Bandpass Applied)', fontsize=11, fontweight='bold')
ax2.set_xlabel('Depth (mm)', fontsize=10)
ax2.set_ylabel('Amplitude', fontsize=10)
ax2.legend()
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()