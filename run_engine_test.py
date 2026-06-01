import numpy as np
import matplotlib.pyplot as plt

# وارد کردن کلاس موتور شبیه‌ساز که در مرحله قبل ساختیم
from ultrasonic_engine import UltrasonicSimulationEngine

# ۱. تعریف خواص فیزیکی قطعه (فرض: فولاد کربنی استاندارد صنعتی)
STEEL_VELOCITY = 5900.0      # سرعت صوت در فولاد (متر بر ثانیه)
STEEL_ATTENUATION = 1.5      # ضریب تضعیف موج در فولاد (نپر بر متر)

# ۲. راه‌اندازی موتور اصلی
engine = UltrasonicSimulationEngine(
    sound_velocity=STEEL_VELOCITY,
    attenuation_coeff=STEEL_ATTENUATION
)


defect_depth = 0.02        
reflection_coeff = 0.5      
probe_freq = 5e6            
bw_factor = 0.5             


time_array, echo_signal = engine.simulate_echo(
    defect_depth=defect_depth,
    reflection_coeff=reflection_coeff,
    probe_freq=probe_freq,
    bw_factor=bw_factor
)


plt.figure(figsize=(10, 4))


plt.plot(time_array * 1e6, echo_signal, color='#b30000', linewidth=1.5, label='Echo Signal')


expected_tof_us = (2 * defect_depth / STEEL_VELOCITY) * 1e6


plt.axvline(x=expected_tof_us, color='green', linestyle='--', linewidth=1.2, 
            label=f'Theoretical ToF: {expected_tof_us:.2f} µs')


plt.title('Simulated A-Scan Echo from a 20mm Defect in Steel', fontsize=12, fontweight='bold')
plt.xlabel('Time (µs)', fontsize=11)
plt.ylabel('Amplitude (Voltage)', fontsize=11)
plt.legend(loc='upper right')
plt.grid(True, linestyle=':', alpha=0.7)
plt.xlim(expected_tof_us - 1.5, expected_tof_us + 1.5) 
plt.tight_layout()


plt.show()