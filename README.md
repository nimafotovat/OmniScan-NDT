# 🔬 OmniScan Pro: Autonomous Ultrasonic NDT & AI Command Center

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.20%2B-FF4B4B?style=for-the-badge&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-Interactive-3F4F75?style=for-the-badge&logo=plotly)
![SciPy](https://img.shields.io/badge/SciPy-Signal_Processing-8CAAE6?style=for-the-badge&logo=scipy)
![License](https://img.shields.io/badge/License-Academic-green?style=for-the-badge)

## 📌 Project Overview
**OmniScan Pro** is an advanced, high-fidelity Ultrasonic Non-Destructive Testing (NDT) simulator and automated evaluation dashboard. Developed to bridge the gap between academic wave physics and industrial AI diagnostics, this system simulates real-time A-Scan acoustic pulse-echo responses within various metallic and non-metallic mediums.

The core engine mathematically models acoustic wave propagation, beam divergence, phase inversion thermodynamics, and frequency-dependent attenuation, providing a hyper-realistic representation of industrial ultrasonic inspections.

## ✨ Key Technical Features

### 🧠 1. AI-Driven Flaw Characterization
- **Automated Disposition:** Continuously monitors RF signals against ASME Distance Amplitude Correction (DAC) reference curves to automatically output **ACCEPTED** or **REJECTED** statuses.
- **Phase Inversion Analysis:** Evaluates the reflection coefficient ($R$) polarity to mathematically distinguish between air-filled cracks (voids) and solid inclusions (slag).

### 📡 2. Advanced Signal Processing Pipeline
- **Hilbert Transform Enveloping:** Extracts the analytical signal envelope for precise peak detection, mirroring enterprise-grade oscilloscope hardware.
- **FFT Spectral Analytics:** Real-time Fast Fourier Transform analysis ensures transducer frequency integrity and validates bandpass filter operations.
- **Digital TGC Integration:** Time-Gain Compensation circuitry simulation to counteract exponential depth attenuation.

### 🛡️ 3. Secure Enterprise Reporting
- **In-Memory PDF Generation:** Compiles tamper-proof, dynamically generated official inspection reports directly in the RAM (bypassing local storage security risks and download manager conflicts).
- **Stealth Export:** Raw CSV data matrices and PDFs are strictly protected and isolated within the session state until manual verification.

---

## ⚙️ Core Physics Engine Mechanics
The simulation is built upon fundamental acoustic principles:
- **Acoustic Impedance ($Z$):** $Z = \rho \times V$
- **Reflection Coefficient ($R$):** $R = \frac{Z_2 - Z_1}{Z_2 + Z_1}$ (Accounts for $180^\circ$ phase shifts when evaluating low-impedance flaws).
- **Near-Field Length ($N$):** $N = \frac{D^2 \times f}{4 \times V}$

---

## 🚀 Installation & Usage

### Prerequisites
Ensure you have Python 3.8+ installed. It is highly recommended to use a virtual environment (`venv`).

### Setup Instructions
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/YourUsername/OmniScan-NDT.git](https://github.com/YourUsername/OmniScan-NDT.git)
   cd OmniScan-NDT
## 📊 Visual Interface
![Dashboard Preview](images/my_dashboard.png)

*Interactive A-Scan Analysis Panel and AI Diagnostic Dashboard.*