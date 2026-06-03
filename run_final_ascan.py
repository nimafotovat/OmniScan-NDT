import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR
from ascan_pipeline import run_ascan_pipeline

V, RHO, ALPHA, THICK, FS = 5900.0, 7800.0, 1.5, 0.10, 100e6
F_PROBE = 5e6

engine = UltrasonicSimulationEngine(V, RHO, ALPHA, THICK, sample_rate=FS)
out = run_ascan_pipeline(
    engine, [(0.02, Z_AIR)], F_PROBE, 0.5, FS, snr_db=20.0,
    use_filter=True, ref_R=0.16,
)
depth_mm = out["depth_array"]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
ax1.plot(depth_mm, out["clean_echo"], color="gray", lw=0.9)
ax1.set_ylabel("RF + noise")
ax1.set_title("Raw noisy A-Scan")
ax1.grid(True, alpha=0.3)

ax2.plot(depth_mm, out["rf_signal"], color="#2E5A88", lw=1.2)
ax2.axvline(20, color="#A93226", ls="--", label="20 mm flaw")
ax2.set_xlabel("Depth (mm)")
ax2.set_ylabel("Filtered RF")
ax2.set_title("Bandpass output (zero-phase filtfilt)")
ax2.legend()
ax2.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
