# Development of a Python-Based Simulator for Ultrasonic Pulse-Echo A-Scan Signal

**Undergraduate final project — Non-Destructive Testing (NDT)**  
**Scope:** one-dimensional pulse-echo **A-Scan only** (no B-Scan, no imaging arrays).

---

## Abstract

A numerical simulator was implemented in Python to model the ultrasonic pulse-echo chain used in contact inspection of homogeneous specimens. The tool generates a synthetic **A-Scan** voltage trace as a function of time or depth by sequentially modeling (i) transducer excitation as a Gaussian-modulated sinusoid, (ii) one-way and round-trip travel time with material velocity and exponential attenuation, (iii) reflection from planar discontinuities through the normal-incidence pressure reflection coefficient, and (iv) receiver processing including additive Gaussian noise and zero-phase bandpass filtering. The interactive dashboard plots the radio-frequency (RF) trace, its Hilbert envelope, and a depth-dependent reference curve for amplitude comparison. Analytical checks verify echo arrival times via \(t = 2d/v\) and peak depths against user-defined reflector locations. The software is intended for parameter studies before laboratory work and as a reproducible baseline for future journal extensions (e.g., probe models, scattering, or 2-D field solvers) without altering the present 1-D scope.

**Keywords:** ultrasonic testing, pulse-echo, A-Scan, acoustic impedance, reflection coefficient, Hilbert envelope, Python simulation

---

## 1. Introduction

Pulse-echo ultrasonics records a one-dimensional time waveform (A-Scan) whose echoes map to reflector depth when longitudinal velocity is known. Undergraduate and early research projects often require a transparent simulator tied to coursework equations rather than a black-box commercial ray tracer. This repository delivers that capability under the formal project title above: strictly **A-Scan**, strictly **pulse-echo**, with optional noise and filtering to emulate laboratory bandwidth and SNR limits.

---

## 2. Physical principles

### 2.1 Gaussian-modulated tone burst

The transmitted particle velocity is approximated as

\[
s(t) = A \, \exp\!\left(-\frac{(t-t_0)^2}{2\sigma^2}\right) \sin(2\pi f_c t),
\qquad
\sigma = \frac{1}{2\pi f_c \,\mathrm{BW}}.
\]

Implementation: `wave_math.generate_gaussian_pulse`.

### 2.2 Time of flight (depth axis)

Round-trip travel time to depth \(d\):

\[
t = \frac{2d}{v}.
\]

Depth axis for plotting: \(d = tv/2\). Implementation: `UltrasonicSimulationEngine.depth_to_tof`.

### 2.3 Attenuation

Echo amplitude is scaled by exponential material loss along the round-trip path \(2d\):

\[
A_{\mathrm{mat}} = \exp(-\alpha(f)\, 2d),
\qquad
\alpha(f) = \alpha_0 \left(\frac{f}{1\,\mathrm{MHz}}\right)^{1.5}.
\]

A simplified beam-spread factor reduces amplitude beyond the near-field distance \(N = D^2 f / (4v)\).

### 2.4 Reflection coefficient (phase preserved)

Normal-incidence pressure reflection coefficient between specimen impedance \(Z_p = \rho v\) and defect impedance \(Z_d\):

\[
R = \frac{Z_d - Z_p}{Z_d + Z_p}.
\]

**No absolute value is applied to \(R\)** when synthesizing RF echoes; negative \(R\) inverts the echo polarity (phase inversion for low-impedance voids). The Hilbert envelope \(|\mathcal{H}\{s(t)\}|\) uses magnitude **only** for peak detection and DAC comparison, which is standard in flaw sizing workflows.

Preset impedances (Rayl): air \(\approx 4\times10^2\), water \(\approx 1.5\times10^6\). Back-wall echo uses a steel–air interface at the far boundary.

### 2.5 Receiver processing

- Additive white Gaussian noise at a user-defined peak SNR (dB).
- Butterworth bandpass, `scipy.signal.filtfilt` (zero phase) over fractional bandwidth about \(f_c\).

---

## 3. Software architecture

| File | Role |
|------|------|
| `wave_math.py` | Gaussian pulse |
| `ultrasonic_engine.py` | Propagation, reflection, DAC reference |
| `signal_processing.py` | Noise, filter, envelope |
| `ascan_pipeline.py` | End-to-end A-Scan assembly |
| `dashboard.py` | Streamlit UI, plots, export |
| `tests/test_physics.py` | ToF, reflection sign, peak depth |
| `run_*.py` | Matplotlib demonstration scripts |

Legacy method names (`simulate_complex_echoes`, `generate_dac_curve`, etc.) remain as aliases for compatibility.

---

## 4. Validation methodology

Validation follows the project proposal: compare simulated results with analytical expectations.

1. **Time law:** compute \(t = 2d/v\) for each programmed reflector and report in the validation table.
2. **Amplitude law:** compute \(A = R \exp(-\alpha 2d)\,B(d)\) with the same \(\alpha\) and beam factor \(B\) used in synthesis; **signed** \(R\) and **signed** theoretical amplitude are listed.
3. **Depth error:** locate the envelope peak in a local depth window around each reflector; report \(\Delta d = d_{\mathrm{meas}} - d_{\mathrm{input}}\).
4. **Automated tests:** `pytest` checks negative \(R\) for air in steel, monotonic decay with depth, and peak depth within 2 mm of input for a steel reference case.
5. **Null case:** empty defect list yields dominant back-wall peak near specimen thickness.

Acceptance in the dashboard is defined as absence of mid-wall envelope peaks above the educational DAC reference curve (not a substitute for code-compliant ASME Section V qualification).

---

## 5. Installation and execution

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
pytest
streamlit run dashboard.py
```

Optional figures: `python run_pulse_test.py`, `run_engine_test.py`, `run_final_ascan.py`, `run_advanced_scenario.py`.

---

## 6. Dashboard guide (defense demonstration)

1. **A-Scan display** — RF (signed), envelope, DAC reference; vertical markers at flaw depths and back-wall.
2. **DAC sensitivity** — indicates whether alternate reference amplitudes would clear detected peaks.
3. **Analytical validation** — table of ToF, signed \(R\), signed theoretical amplitude, measured peak depth, depth error; FFT of processed RF.
4. **Report export** — PDF summary and CSV time series.

Recommended defense scenario: steel, 70 mm thickness, flaws at 25 mm and 50 mm (air + water fills), 5 MHz probe, SNR 30 dB, filter enabled.

---

## 7. Results and discussion (template for thesis / ISI draft)

- Echo arrival depths align with \(t=2d/v\) within discretization error (see validation table metric *Max depth error*).
- Air-filled reflectors produce \(R \approx -1\) and inverted RF polarity; water in steel produces weaker negative \(R\) but still phase-inverted relative to a high-impedance inclusion.
- Bandpass filtering preserves ToF (zero-phase `filtfilt`) while concentrating spectral energy near the probe center frequency.
- Limitations: 1-D superposition, no mode conversion, no near-surface creep, no true ASME DAC block calibration, educational DAC reference only.

---

## 8. Conclusion

The simulator fulfills the approved proposal scope: Python implementation of a **pulse-echo A-Scan** simulator with Gaussian pulse generation, velocity-based propagation, exponential attenuation, reflection-coefficient echoes with preserved phase, noise and filtering, and depth-domain visualization with analytical validation. The codebase is structured for undergraduate defense and can anchor a journal paper introduction and methods section, while higher-dimensional imaging (B-Scan, C-Scan, FEM) is explicitly out of scope here.

---

## 9. Future work (journal extension path)

- Laboratory calibration of \(\alpha_0\) and bandwidth from reference echoes.
- Frequency-dependent probe aperture model.
- Stochastic grain noise and repeatable Monte Carlo receiver studies.
- Coupled 2-D wave solvers (outside current repository scope).

---

## References (indicative)

1. Krautkramer, J. & Krautkramer, H., *Ultrasonic Testing of Materials*, Springer.
2. Schmerr, L., *Fundamentals of Ultrasonic Nondestructive Evaluation*, Plenum.
3. ASTM E1316 (terminology) and practice documents for pulse-echo examination (for laboratory comparison only).

---

## License

Academic use. Cite the project title and repository URL in thesis and manuscript acknowledgments.
