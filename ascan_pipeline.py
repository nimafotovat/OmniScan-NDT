import numpy as np

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR, Z_WATER
from signal_processing import add_gaussian_noise, apply_bandpass_filter, compute_envelope

FILLER_TO_Z = {
    "Air Void (Crack)": Z_AIR,
    "Water Inclusion": Z_WATER,
}


def filler_to_impedance(label):
    return FILLER_TO_Z[label]


def min_peak_distance_samples(velocity, fs, min_sep_mm=2.0):
    d_m = min_sep_mm / 1000.0
    return max(int((2 * d_m / velocity) * fs), 20)


def classify_reflector(engine, z_defect):
    if z_defect <= Z_AIR * 10:
        return "Air void"
    if z_defect < engine.Z_part * 0.05:
        return "Low-Z inclusion"
    if engine.reflection_coeff(z_defect) > 0:
        return "High-Z inclusion"
    return "Intermediate reflector"


def match_nearest_defect(depth_mm, defects):
    best, err_min = None, float("inf")
    for depth_m, z in defects:
        err = abs(depth_mm - depth_m * 1000.0)
        if err < err_min:
            err_min, best = err, (depth_m, z)
    if best is None or err_min > 3.0:
        return None
    return best


def run_ascan_pipeline(
    engine,
    defects,
    probe_freq_hz,
    bw_factor,
    fs,
    snr_db,
    use_filter,
    ref_R,
    enable_tgc=False,
    tgc_slope=0.0,
    filter_bandwidth=0.5,
    filter_order=5,
):
    time_array, clean = engine.simulate_ascan(defects, probe_freq_hz, bw_factor)
    noisy = add_gaussian_noise(clean, snr_db=snr_db)
    if use_filter:
        rf = apply_bandpass_filter(noisy, fs, probe_freq_hz, filter_bandwidth, filter_order)
    else:
        rf = noisy.copy()

    depth_mm = (time_array * engine.velocity / 2) * 1000
    dac = engine.dac_curve(time_array, probe_freq_hz, ref_R)
    tgc = np.ones_like(depth_mm)
    if enable_tgc:
        tgc = 10 ** ((tgc_slope * depth_mm) / 20)
        rf *= tgc
        dac *= tgc

    envelope = compute_envelope(rf)
    return {
        "time_array": time_array,
        "clean_echo": clean,
        "rf_signal": rf,
        "depth_array": depth_mm,
        "dac_curve": dac,
        "signal_envelope": envelope,
        "tgc_gain": tgc,
    }
