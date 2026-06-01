import numpy as np
import matplotlib.pyplot as plt

# وارد کردن تابع از فایل جدید
from wave_math import generate_gaussian_pulse

# تنظیمات اولیه زمان شبیه‌سازی
sample_rate = 100e6  
time_array = np.arange(0, 5e-6, 1/sample_rate)  

# مشخصات پروب فراصوت
f_c = 5e6           
bw_factor = 0.5     
t_delay = 2e-6      

# فراخوانی تابع تولید پالس
pulse_signal = generate_gaussian_pulse(
    time_array=time_array, 
    center_freq=f_c, 
    bandwidth_factor=bw_factor, 
    time_delay=t_delay
)

# تنظیمات رسم نمودار
plt.figure(figsize=(10, 4))
plt.plot(time_array * 1e6, pulse_signal, color='#003366', linewidth=1.5)

plt.title('Simulated Ultrasonic Transducer Pulse', fontsize=12, fontweight='bold')
plt.xlabel('Time (µs)', fontsize=11)
plt.ylabel('Normalized Amplitude', fontsize=11)
plt.axhline(0, color='black', linewidth=0.8, linestyle='-')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()

# نمایش خروجی
plt.show()