import numpy as np
import pytest

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR, Z_WATER
from ascan_pipeline import run_ascan_pipeline, filler_to_impedance

STEEL_V = 5900.0
STEEL_RHO = 7800.0
STEEL_ALPHA = 1.5
THICKNESS = 0.05
FS = 100e6
PROBE_F = 5e6


@pytest.fixture
def engine():
    return UltrasonicSimulationEngine(
        STEEL_V, STEEL_RHO, STEEL_ALPHA, THICKNESS, 0.01, FS
    )


def test_reflection_sign(engine):
    r_air = engine.reflection_coeff(Z_AIR)
    r_water = engine.reflection_coeff(Z_WATER)
    assert r_air < -0.9
    assert r_water < 0
    assert abs(r_air) > abs(r_water)


def test_tof(engine):
    d = 0.02
    assert abs(engine.depth_to_tof(d) - 2 * d / STEEL_V) < 1e-12


def test_attenuation_with_depth(engine):
    r = 0.2
    a1 = engine.expected_echo_amplitude(0.01, PROBE_F, r)
    a2 = engine.expected_echo_amplitude(0.04, PROBE_F, r)
    assert abs(a1) > abs(a2)


def test_flaw_peak_depth(engine):
    d_m = 0.015
    t, echo = engine.simulate_ascan([(d_m, Z_AIR)], PROBE_F, 0.5)
    depth_mm = (t * STEEL_V / 2) * 1000
    i = np.argmax(np.abs(echo))
    assert abs(depth_mm[i] - d_m * 1000) < 1.5


def test_backwall_only(engine):
    t, echo = engine.simulate_ascan([], PROBE_F, 0.5)
    depth_mm = (t * STEEL_V / 2) * 1000
    i = np.argmax(np.abs(echo))
    assert abs(depth_mm[i] - THICKNESS * 1000) < 2.0


def test_pipeline_tof(engine):
    out = run_ascan_pipeline(
        engine, [(0.02, Z_AIR)], PROBE_F, 0.5, FS, 40.0, True, 0.16
    )
    i = np.argmax(out["signal_envelope"])
    assert abs(out["depth_array"][i] - 20.0) < 2.0


def test_fillers():
    assert filler_to_impedance("Air Void (Crack)") == Z_AIR
    assert filler_to_impedance("Water Inclusion") == Z_WATER
