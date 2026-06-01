import numpy as np
from wave_math import generate_gaussian_pulse

class UltrasonicSimulationEngine:
    """
    موتور هوشمند شبیه‌سازی فراصوت
    شامل: امپدانس آکوستیک، وارونگی فاز، پخشیدگی پرتو، تضعیف فرکانسی و تولید منحنی DAC
    """
    def __init__(self, sound_velocity: float, material_density: float, attenuation_coeff: float, part_thickness: float, probe_diameter: float = 0.01, sample_rate: float = 100e6):
        self.velocity = sound_velocity      
        self.density = material_density          
        self.Z_part = self.velocity * self.density 
        self.base_alpha = attenuation_coeff      
        self.thickness = part_thickness     
        self.probe_diameter = probe_diameter     
        self.sample_rate = sample_rate

    def calculate_effective_attenuation(self, probe_freq: float) -> float:
        f_mhz = probe_freq / 1e6 
        power_factor = 1.5       
        return self.base_alpha * (f_mhz ** power_factor)

    def calculate_reflection_coeff(self, defect_impedance: float) -> float:
        """
        وارونگی فاز (Phase Inversion): 
        حذف قدر مطلق باعث می‌شود اگر امپدانس عیب کمتر از قطعه باشد (مثل هوا در فولاد)، 
        ضریب منفی شده و سیگنال ۱۸۰ درجه تغییر فاز دهد.
        """
        R = (defect_impedance - self.Z_part) / (defect_impedance + self.Z_part)
        return R

    def calculate_beam_spread_loss(self, distance: float, probe_freq: float) -> float:
        near_field_length = (self.probe_diameter ** 2 * probe_freq) / (4 * self.velocity)
        if distance > near_field_length:
            return near_field_length / distance
        return 1.0 

    def simulate_complex_echoes(self, defects: list, probe_freq: float, bw_factor: float):
        max_time_of_flight = (2 * self.thickness) / self.velocity
        total_time = max_time_of_flight + (20 / probe_freq) 
        time_array = np.arange(0, total_time, 1 / self.sample_rate)
        
        total_signal = np.zeros_like(time_array)
        alpha_f = self.calculate_effective_attenuation(probe_freq)
        
       
        for depth, defect_impedance in defects: 
            if depth >= self.thickness: continue 
            tof = (2 * depth) / self.velocity
            material_loss = np.exp(-alpha_f * (2 * depth))
            beam_loss = self.calculate_beam_spread_loss(2 * depth, probe_freq)
            R = self.calculate_reflection_coeff(defect_impedance)
            amplitude = R * material_loss * beam_loss
            
            pulse = generate_gaussian_pulse(time_array, probe_freq, bw_factor, tof, amplitude)
            total_signal += pulse
            
       
        Z_air = 400.0 
        R_backwall = self.calculate_reflection_coeff(Z_air)
        bw_tof = (2 * self.thickness) / self.velocity
        bw_material_loss = np.exp(-alpha_f * (2 * self.thickness))
        bw_beam_loss = self.calculate_beam_spread_loss(2 * self.thickness, probe_freq)
        
        bw_amplitude = R_backwall * bw_material_loss * bw_beam_loss
        bw_pulse = generate_gaussian_pulse(time_array, probe_freq, bw_factor, bw_tof, bw_amplitude)
        total_signal += bw_pulse
        
        return time_array, total_signal

    def generate_dac_curve(self, time_array: np.ndarray, probe_freq: float, reference_R: float = 0.2) -> np.ndarray:
        """تولید منحنی استاندارد DAC بر اساس یک رفلکتور مرجع (مثلاً سوراخ 2 میلی‌متری)"""
        depth_array = (time_array * self.velocity) / 2
        alpha_f = self.calculate_effective_attenuation(probe_freq)
        
        dac_curve = np.zeros_like(depth_array)
        for i, d in enumerate(depth_array):
            if d <= 0.001: 
                dac_curve[i] = reference_R
                continue
            material_loss = np.exp(-alpha_f * (2 * d))
            beam_loss = self.calculate_beam_spread_loss(2 * d, probe_freq)
            dac_curve[i] = reference_R * material_loss * beam_loss
            
        return dac_curve