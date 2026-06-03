import matplotlib.pyplot as plt

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR, Z_WATER
from ascan_pipeline import run_ascan_pipeline

V, RHO, ALPHA, THICK, FS = 5900.0, 7800.0, 1.5, 0.05, 100e6
F_PROBE = 5e6

engine = UltrasonicSimulationEngine(V, RHO, ALPHA, THICK, sample_rate=FS)
defects = [(0.015, Z_AIR), (0.035, Z_WATER)]
out = run_ascan_pipeline(
    engine, defects, F_PROBE, 0.5, FS, 18.0, True, 0.16,
    filter_bandwidth=0.6, filter_order=4,
)
depth_mm = out["depth_array"]

plt.figure(figsize=(11, 4.5))
plt.plot(depth_mm, out["rf_signal"], color="#2E5A88", lw=1.3)
for x, lbl in [(15, "Flaw 1"), (35, "Flaw 2"), (50, "Back-wall")]:
    plt.axvline(x, ls="--", lw=1, label=lbl)
plt.xlabel("Depth (mm)")
plt.ylabel("Amplitude")
plt.title("A-Scan — two reflectors + back-wall echo")
plt.xlim(0, 55)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
