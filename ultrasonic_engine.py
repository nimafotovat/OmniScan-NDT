import numpy as np
from wave_math import generate_gaussian_pulse

Z_AIR = 400.0
Z_WATER = 1.48e6


class UltrasonicSimulationEngine:
    def __init__(
        self,
        sound_velocity,
        material_density,
        attenuation_coeff,
        part_thickness,
        probe_diameter=0.01,
        sample_rate=100e6,
    ):
        self.velocity = sound_velocity
        self.density = material_density
        self.Z_part = self.velocity * self.density
        self.base_alpha = attenuation_coeff
        self.thickness = part_thickness
        self.probe_diameter = probe_diameter
        self.sample_rate = sample_rate

    def effective_attenuation(self, probe_freq):
        f_mhz = probe_freq / 1e6
        return self.base_alpha * (f_mhz ** 1.5)

    def reflection_coeff(self, defect_impedance):
        return (defect_impedance - self.Z_part) / (defect_impedance + self.Z_part)

    def near_field_length(self, probe_freq):
        return (self.probe_diameter ** 2 * probe_freq) / (4 * self.velocity)

    def depth_to_tof(self, depth_m):
        return (2 * depth_m) / self.velocity

    def expected_echo_amplitude(self, depth_m, probe_freq, reflection_R):
        alpha = self.effective_attenuation(probe_freq)
        path = 2 * depth_m
        atten = np.exp(-alpha * path)
        beam = self.beam_spread_loss(path, probe_freq)
        return reflection_R * atten * beam

    def beam_spread_loss(self, distance, probe_freq):
        n = self.near_field_length(probe_freq)
        if distance > n:
            return n / distance
        return 1.0

    def simulate_ascan(self, defects, probe_freq, bw_factor):
        t_max = (2 * self.thickness) / self.velocity + (20 / probe_freq)
        time_array = np.arange(0, t_max, 1 / self.sample_rate)
        signal = np.zeros_like(time_array)
        alpha = self.effective_attenuation(probe_freq)

        for depth, z_defect in defects:
            if depth >= self.thickness:
                continue
            tof = self.depth_to_tof(depth)
            atten = np.exp(-alpha * (2 * depth))
            beam = self.beam_spread_loss(2 * depth, probe_freq)
            r = self.reflection_coeff(z_defect)
            amp = r * atten * beam
            signal += generate_gaussian_pulse(time_array, probe_freq, bw_factor, tof, amp)

        r_bw = self.reflection_coeff(Z_AIR)
        tof_bw = self.depth_to_tof(self.thickness)
        atten_bw = np.exp(-alpha * (2 * self.thickness))
        beam_bw = self.beam_spread_loss(2 * self.thickness, probe_freq)
        amp_bw = r_bw * atten_bw * beam_bw
        signal += generate_gaussian_pulse(time_array, probe_freq, bw_factor, tof_bw, amp_bw)

        return time_array, signal

    def dac_curve(self, time_array, probe_freq, reference_R=0.2):
        depth = (time_array * self.velocity) / 2
        out = np.zeros_like(depth, dtype=float)
        for i, d in enumerate(depth):
            out[i] = self.expected_echo_amplitude(d, probe_freq, reference_R)
        return out

    # aliases used by older call sites
    calculate_effective_attenuation = effective_attenuation
    calculate_reflection_coeff = reflection_coeff
    calculate_beam_spread_loss = beam_spread_loss
    simulate_complex_echoes = simulate_ascan
    generate_dac_curve = dac_curve
