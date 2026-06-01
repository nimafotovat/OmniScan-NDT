import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_engine import UltrasonicSimulationEngine
from signal_processing import add_gaussian_noise, apply_bandpass_filter


STEEL_VELOCITY = 5900.0      
STEEL_ATTENUATION = 1.5      
PART_THICKNESS = 0.05        
SAMPLE_RATE = 100e6
PROBE_FREQ = 5e6

engine = UltrasonicSimulationEngine(
    sound_velocity=STEEL_VELOCITY, 
    attenuation_coeff=STEEL_ATTENUATION, 
    part_thickness=PART_THICKNESS,
    sample_rate=SAMPLE_RATE
)


defects_list = [
    (0.015, 0.4), 
    (0.035, 0.6)
]


time_array, complex_echo = engine.simulate_complex_echoes(
    defects=defects_list, 
    probe_freq=PROBE_FREQ, 
    bw_factor=0.5
)


noisy_signal = add_gaussian_noise(complex_echo, snr_db=18.0)
final_signal = apply_bandpass_filter(noisy_signal, 2e6, 8e6, SAMPLE_RATE, order=4)


depth_array_mm = (time_array * STEEL_VELOCITY / 2) * 1000


plt.figure(figsize=(12, 5))
plt.plot(depth_array_mm, final_signal, color='#002244', linewidth=1.5, label='Processed A-Scan')


plt.axvline(x=15, color='orange', linestyle='--', label='Defect 1 (15mm)')
plt.axvline(x=35, color='red', linestyle='--', label='Defect 2 (35mm)')
plt.axvline(x=50, color='green', linestyle='--', linewidth=2, label='Back-wall (50mm)')

plt.title('Advanced NDT Simulation: Multiple Defects and Back-wall Echo', fontsize=14, fontweight='bold')
plt.xlabel('Depth in Steel (mm)', fontsize=12)
plt.ylabel('Signal Amplitude', fontsize=12)
plt.xlim(0, 55) 
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()

plt.show()