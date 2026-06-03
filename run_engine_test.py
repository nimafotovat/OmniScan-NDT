import numpy as np
import matplotlib.pyplot as plt

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR

V = 5900.0
RHO = 7800.0
ALPHA = 1.5
THICK = 0.10
FS = 100e6
F_PROBE = 5e6

engine = UltrasonicSimulationEngine(V, RHO, ALPHA, THICK, sample_rate=FS)
t, echo = engine.simulate_ascan([(0.02, Z_AIR)], F_PROBE, 0.5)
depth_mm = (t * V / 2) * 1000

plt.figure(figsize=(10, 4))
plt.plot(depth_mm, echo, color="#2E5A88", lw=1.2, label="A-Scan RF")
plt.axvline(20, color="#117A65", ls="--", label="Flaw @ 20 mm")
plt.axvline(THICK * 1000, color="#1B4F72", ls="-.", label="Back-wall")
plt.xlabel("Depth (mm)")
plt.ylabel("Amplitude")
plt.title("Pulse-echo A-Scan — steel specimen")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
