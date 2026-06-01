import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks, hilbert
from scipy.fft import fft, fftfreq
from fpdf import FPDF
from datetime import datetime

from ultrasonic_engine import UltrasonicSimulationEngine
from signal_processing import add_gaussian_noise, apply_bandpass_filter


st.set_page_config(page_title="OmniScan A-Scan | Expert NDT", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .cyber-title { font-size: 2.6rem; font-weight: 900; background: -webkit-linear-gradient(45deg, #00FFCC, #0077FF); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 0px;}
    .sub-text { font-size: 1.1rem; color: #8A9BA8; margin-bottom: 25px;}
    .metric-card { background: #11141A; padding: 18px; border-radius: 12px; border: 1px solid #2B3040; border-top: 4px solid #00FFCC; text-align: center; box-shadow: 0 6px 12px rgba(0,0,0,0.4); }
    .value-text { font-size: 22px; font-weight: bold; color: #FFFFFF; }
    .label-text { font-size: 12px; color: #A0AEC0; text-transform: uppercase; letter-spacing: 1px;}
    .ai-box-danger { background: rgba(255, 51, 102, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #FF3366; margin-bottom: 10px; }
    .ai-box-success { background: rgba(0, 255, 204, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #00FFCC; margin-bottom: 10px; }
    .badge-danger { background-color: #FF3366; color: white; padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: bold; }
    .badge-info { background-color: #0077FF; color: white; padding: 3px 8px; border-radius: 5px; font-size: 11px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="cyber-title">🌐 Autonomous A-Scan Command Center</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">Interactive Digital Oscilloscope, AI Flaw Characterization & Secure PDF Reporting</div>', unsafe_allow_html=True)
st.markdown("---")


st.sidebar.header("🎛️ Inspection Parameters")

st.sidebar.subheader("1. Metallurgy Profile")
material_choice = st.sidebar.selectbox("Material Preset", ["Steel (Carbon)", "Aluminum (6061)", "Copper (Pure)", "PVC Plastic", "Custom Input"])

if material_choice == "Custom Input":
    velocity = st.sidebar.number_input("Velocity (m/s)", 1000.0, 10000.0, 5900.0, 50.0)
    density = st.sidebar.number_input("Density (kg/m³)", 500.0, 20000.0, 7800.0, 50.0)
    base_alpha = st.sidebar.number_input("Attenuation (Np/m)", 0.1, 20.0, 1.5, 0.1)
else:
    presets = {
        "Steel (Carbon)": (5900.0, 7800.0, 1.5), 
        "Aluminum (6061)": (6320.0, 2700.0, 1.2), 
        "Copper (Pure)": (4700.0, 8960.0, 2.0), 
        "PVC Plastic": (2395.0, 1380.0, 4.5)
    }
    velocity, density, base_alpha = presets[material_choice]
    st.sidebar.info(f"**Properties:**\nVel: {velocity} m/s | ρ: {density} kg/m³")

thickness_mm = st.sidebar.slider("Part Thickness (mm)", 15.0, 120.0, 70.0)

st.sidebar.subheader("2. Flaw Simulator")
defect1_depth_mm = st.sidebar.slider("Flaw 1 Depth (mm)", 3.0, thickness_mm-3.0, 25.0)
defect2_depth_mm = st.sidebar.slider("Flaw 2 Depth (mm)", 3.0, thickness_mm-3.0, 50.0)
defect_filler = st.sidebar.radio("Defect Core Medium:", ["Air Void (Crack)", "Water Inclusion (Slag)"])
defect_Z = 400.0 if defect_filler == "Air Void (Crack)" else 1.48e6

st.sidebar.subheader("3. Transducer Electronics")
probe_freq_mhz = st.sidebar.slider("Probe Frequency (MHz)", 1.0, 10.0, 5.0, 0.5)
dac_ref_size = st.sidebar.selectbox("ASME Calibration Standard", ["Ø 1.0mm (Strict)", "Ø 2.0mm (Standard)", "Ø 3.0mm (Relaxed)"])
ref_R = {"Ø 1.0mm (Strict)": 0.08, "Ø 2.0mm (Standard)": 0.16, "Ø 3.0mm (Relaxed)": 0.28}[dac_ref_size]

enable_tgc = st.sidebar.checkbox("Enable Digital TGC Gain", value=False)
tgc_slope = st.sidebar.slider("TGC Slope (dB/mm)", 0.0, 2.5, 0.5, disabled=not enable_tgc)


st.sidebar.subheader("4. Signal Processing (Proposal)")
snr_db = st.sidebar.slider("Signal-to-Noise Ratio (SNR dB)", 10.0, 50.0, 30.0, 1.0)
use_filter = st.sidebar.checkbox("Apply Transducer Bandpass Filter", value=True)


probe_freq_hz = probe_freq_mhz * 1e6
fs = 100e6  # Sampling frequency
engine = UltrasonicSimulationEngine(velocity, density, base_alpha, thickness_mm/1000.0, 0.01, fs)
defects = [(defect1_depth_mm/1000.0, defect_Z), (defect2_depth_mm/1000.0, defect_Z)]

time_array, complex_echo = engine.simulate_complex_echoes(defects, probe_freq_hz, 0.5)


noisy_signal = add_gaussian_noise(complex_echo, snr_db=snr_db)
if use_filter:
    rf_signal = apply_bandpass_filter(noisy_signal, fs, center_freq=probe_freq_hz, bandwidth_percent=0.5, order=5)
else:
    rf_signal = noisy_signal

depth_array = (time_array * velocity / 2) * 1000
dac_curve = engine.generate_dac_curve(time_array, probe_freq_hz, ref_R)

if enable_tgc:
    tgc_gain = 10 ** ((tgc_slope * depth_array) / 20)
    rf_signal *= tgc_gain
    dac_curve *= tgc_gain

analytic_signal = hilbert(rf_signal)
signal_envelope = np.abs(analytic_signal)
r_coeff = engine.calculate_reflection_coeff(defect_Z)

# AI Defect Identification
peaks_indices, _ = find_peaks(signal_envelope, height=dac_curve, distance=500)
critical_defects = []
for i in peaks_indices:
    d_mm = depth_array[i]
    if 3.0 < d_mm < (thickness_mm - 2.0):
        if r_coeff < 0:
            flaw_type = "Air-Filled Crack / Void"
            severity = "CRITICAL (Severe Reflection)"
        else:
            flaw_type = "Solid Slag / Inclusion"
            severity = "WARNING (Medium Reflection)"
            
        location_cat = "Near-Surface" if d_mm < 10.0 else "Near Back-wall" if d_mm > (thickness_mm - 10.0) else "Mid-Core"
            
        critical_defects.append({
            "depth": round(d_mm, 2), "amp": round(signal_envelope[i], 3), 
            "limit": round(dac_curve[i], 3), "type": flaw_type, "location": location_cat
        })

is_rejected = len(critical_defects) > 0


m1, m2, m3, m4 = st.columns(4)
m1.markdown(f'<div class="metric-card"><div class="label-text">Acoustic Impedance</div><div class="value-text">{(velocity*density)/1e6:.2f} MRayl</div></div>', unsafe_allow_html=True)
m2.markdown(f'<div class="metric-card"><div class="label-text">Wavelength (λ)</div><div class="value-text">{(velocity/probe_freq_hz)*1000:.3f} mm</div></div>', unsafe_allow_html=True)

if is_rejected:
    m3.markdown('<div class="metric-card" style="border-top: 4px solid #FF3366;"><div class="label-text">AI Disposition</div><div class="value-text" style="color:#FF3366;">🚨 REJECTED</div></div>', unsafe_allow_html=True)
    m4.markdown(f'<div class="metric-card" style="border-top: 4px solid #FF3366;"><div class="label-text">Critical Flaws</div><div class="value-text" style="color:#FF3366;">{len(critical_defects)} Detected</div></div>', unsafe_allow_html=True)
else:
    m3.markdown('<div class="metric-card" style="border-top: 4px solid #00FFCC;"><div class="label-text">AI Disposition</div><div class="value-text" style="color:#00FFCC;">✅ CERTIFIED</div></div>', unsafe_allow_html=True)
    m4.markdown('<div class="metric-card" style="border-top: 4px solid #00FFCC;"><div class="label-text">Critical Flaws</div><div class="value-text" style="color:#00FFCC;">0 Detected</div></div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


current_params = (velocity, density, base_alpha, thickness_mm, defect1_depth_mm, defect2_depth_mm, defect_filler, probe_freq_mhz, dac_ref_size, enable_tgc, tgc_slope, snr_db, use_filter)

if "prev_params" not in st.session_state:
    st.session_state.prev_params = current_params
    st.session_state.export_ready = False


if st.session_state.prev_params != current_params:
    st.session_state.export_ready = False
    st.session_state.prev_params = current_params


tab1, tab2, tab3, tab4 = st.tabs(["📟 AI Diagnostic Oscilloscope", "🛠️ Engineering Recalibration", "🔬 Spectral Physics Lab", "📑 Secure Reporting"])

with tab1:
    graph_side, ai_side = st.columns([1.8, 1.2])
    
    with graph_side:
        st.markdown("**Interactive Plotly A-Scan Matrix**")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=depth_array, y=rf_signal, mode='lines', name='RF Waveform', line=dict(color='#00FFCC', width=1), opacity=0.4))
        fig.add_trace(go.Scatter(x=depth_array, y=signal_envelope, mode='lines', name='Hilbert Envelope', line=dict(color='#B366FF', width=2.5)))
        fig.add_trace(go.Scatter(x=depth_array, y=dac_curve, mode='lines', name='ASME DAC Threshold', line=dict(color='#FF3366', width=2, dash='dash')))
        fig.add_trace(go.Scatter(x=depth_array, y=-dac_curve, mode='lines', name='DAC Bottom', line=dict(color='#FF3366', width=2, dash='dash'), showlegend=False))
        
        fig.update_layout(
            template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117',
            xaxis_title="Depth into Object (mm)", yaxis_title="Voltage Amplitude (V)",
            hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), height=480,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        fig.add_vline(x=thickness_mm, line_width=2, line_dash="dashdot", line_color="#33FF33", annotation_text="Back-wall Boundary")
        st.plotly_chart(fig, use_container_width=True)
        
    with ai_side:
        st.markdown("**🧠 Autonomous Flaw Characterization**")
        if is_rejected:
            for idx, defect in enumerate(critical_defects):
                st.markdown(f"""
                <div class="ai-box-danger">
                    <span class="badge-danger">DEFECT #{idx+1} IDENTIFIED</span><br>
                    <p style="margin-top:8px; margin-bottom:4px;">🎯 <b>Location:</b> {defect['depth']} mm ({defect['location']})</p>
                    <p style="margin-top:0px; margin-bottom:4px;">🔬 <b>Type:</b> <span style="color:#FF3366; font-weight:bold;">{defect['type']}</span></p>
                    <p style="margin-top:0px; margin-bottom:0px;">📉 <b>Signal:</b> {defect['amp']}V (Threshold: {defect['limit']}V)</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ai-box-success">
                <span class="badge-info">SYSTEM ONLINE</span><br>
                <p style="margin-top:8px; color:#00FFCC; font-weight:bold;">Signal Integrity Clear. No discontinuities exceed the ASME DAC curve.</p>
            </div>
            """, unsafe_allow_html=True)

with tab2:
    st.subheader("Automated Parameter Optimization Engine")
    if is_rejected:
        compatible_standard = "None"
        for test_std, test_R in [("Ø 2.0mm (Standard)", 0.16), ("Ø 3.0mm (Relaxed)", 0.28)]:
            test_dac = engine.generate_dac_curve(time_array, probe_freq_hz, test_R)
            if enable_tgc: test_dac *= tgc_gain
            if all(signal_envelope[np.argmin(np.abs(depth_array - d["depth"]))] <= test_dac[np.argmin(np.abs(depth_array - d["depth"]))] for d in critical_defects):
                compatible_standard = test_std
                break
                
        if compatible_standard != "None":
            st.success(f"⚙️ **System Recommendation:** Change ASME Reference Standard to **{compatible_standard}** to successfully certify this component under relaxed guidelines.")
        else:
            st.error("⚙️ **System Recommendation:** Flaws are structurally severe. Component must be sent for mechanical repair (excavation).")
    else:
        st.success("Component is perfectly calibrated and accepted. No recalibration needed.")

with tab3:
    st.subheader("Frequency Domain Analytics & Physics Metadata")
    near_field_mm = ((0.01 ** 2) * probe_freq_hz) / (4 * velocity) * 1000
    alpha_eff_np = engine.calculate_effective_attenuation(probe_freq_hz)
    
    p_col1, p_col2, p_col3 = st.columns(3)
    p_col1.metric("Near-Field Zone (N)", f"{near_field_mm:.2f} mm")
    p_col2.metric("Effective Attenuation (α)", f"{alpha_eff_np:.3f} Np/m")
    p_col3.metric("Reflection Energy Coefficient", f"{r_coeff:.4f}")
    

    st.markdown("### 🧮 Analytical Validation ($t = 2d/v$ & Exponential Attenuation)")
    
    # 1. محاسبه زمان تحلیلی (Theoretical Time)
    t_flaw1_us = (2 * (defect1_depth_mm / 1000.0) / velocity) * 1e6
    t_flaw2_us = (2 * (defect2_depth_mm / 1000.0) / velocity) * 1e6
    t_bwe_us = (2 * (thickness_mm / 1000.0) / velocity) * 1e6
    
  
  
    A0 = 1.0  
    amp_flaw1 = A0 * np.exp(-alpha_eff_np * (2 * (defect1_depth_mm / 1000.0))) * abs(r_coeff)
    amp_flaw2 = A0 * np.exp(-alpha_eff_np * (2 * (defect2_depth_mm / 1000.0))) * abs(r_coeff)
   
    amp_bwe = A0 * np.exp(-alpha_eff_np * (2 * (thickness_mm / 1000.0))) * 1.0 
    
    val_data = {
        "Reflector Target": ["Flaw 1", "Flaw 2", "Back-wall (BWE)"],
        "Depth 'd' (mm)": [defect1_depth_mm, defect2_depth_mm, thickness_mm],
        "Theoretical ToF 't' (µs)": [f"{t_flaw1_us:.3f}", f"{t_flaw2_us:.3f}", f"{t_bwe_us:.3f}"],
        "Expected Amplitude (A)": [f"{amp_flaw1:.4f} V", f"{amp_flaw2:.4f} V", f"{amp_bwe:.4f} V"]
    }
    st.table(pd.DataFrame(val_data))
    

    N = len(rf_signal)
    yf = fft(rf_signal)
    xf = fftfreq(N, 1/100e6)
    pos_xf = xf[:N//2] / 1e6
    pos_yf = np.abs(yf[:N//2])
    
    fig_fft = go.Figure()
    fig_fft.add_trace(go.Scatter(x=pos_xf, y=pos_yf, mode='lines', fill='tozeroy', name='FFT Density', line=dict(color='#B366FF', width=2), fillcolor='rgba(179,102,255,0.15)'))
    fig_fft.update_layout(template="plotly_dark", plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', xaxis_title="Frequency (MHz)", yaxis_title="Spectral Power", xaxis=dict(range=[0, probe_freq_mhz*2.2]), height=300, margin=dict(l=0,r=0,t=10,b=0))
    st.plotly_chart(fig_fft, use_container_width=True)


with tab4:
    st.markdown("### 📑 Secure Inspection Report Generator")
    st.info("💡 **Stealth Mode Active:** Files are locked and hidden from download managers until you click the Compile button.")
    
    def generate_pdf_in_memory():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", 'B', 16)
        pdf.cell(0, 10, "OFFICIAL ULTRASONIC NDT INSPECTION REPORT", ln=True, align='C')
        pdf.set_font("Arial", 'I', 10)
        pdf.cell(0, 8, f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True, align='C')
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.set_fill_color(220, 220, 220)
        pdf.cell(0, 8, " 1. Material & Component Specification", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f" Material Name: {material_choice}", ln=True)
        pdf.cell(0, 8, f" Acoustic Velocity: {velocity} m/s", ln=True)
        pdf.cell(0, 8, f" Material Density: {density} kg/m^3", ln=True)
        pdf.cell(0, 8, f" Tested Thickness: {thickness_mm} mm", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, " 2. Transducer & Calibration Parameters", ln=True, fill=True)
        pdf.set_font("Arial", '', 11)
        pdf.cell(0, 8, f" Center Frequency: {probe_freq_mhz} MHz", ln=True)
        pdf.cell(0, 8, f" Hardware TGC Enabled: {'YES' if enable_tgc else 'NO'} (Slope: {tgc_slope} dB/mm)", ln=True)
        pdf.cell(0, 8, f" Acceptance Standard: ASME {dac_ref_size}", ln=True)
        pdf.ln(5)
        
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, " 3. Automated Defect Evaluation Log", ln=True, fill=True)
        pdf.set_font("Arial", '', 10)
        if not is_rejected:
            pdf.cell(0, 8, " No critical flaws detected. Component meets all structural requirements.", ln=True)
        else:
            pdf.cell(40, 8, "Depth (mm)", border=1, align='C')
            pdf.cell(40, 8, "Peak Amp (V)", border=1, align='C')
            pdf.cell(40, 8, "DAC Limit (V)", border=1, align='C')
            pdf.cell(60, 8, "AI Classification", border=1, align='C')
            pdf.ln()
            for flaw in critical_defects:
                pdf.cell(40, 8, str(flaw["depth"]), border=1, align='C')
                pdf.cell(40, 8, str(flaw["amp"]), border=1, align='C')
                pdf.cell(40, 8, str(flaw["limit"]), border=1, align='C')
                pdf.cell(60, 8, str(flaw["type"][:20]), border=1, align='C')
                pdf.ln()
        pdf.ln(10)
        
        pdf.set_font("Arial", 'B', 14)
        if is_rejected:
            pdf.set_text_color(200, 0, 0)
            pdf.cell(0, 10, "FINAL DISPOSITION: REJECTED", ln=True, align='C')
        else:
            pdf.set_text_color(0, 150, 0)
            pdf.cell(0, 10, "FINAL DISPOSITION: ACCEPTED", ln=True, align='C')
            
        return pdf.output(dest='S').encode('latin-1')

  
    if not st.session_state.export_ready:
        if st.button("⚙️ Compile & Build PDF Report", use_container_width=True):
            st.session_state.pdf_bytes = generate_pdf_in_memory()
            st.session_state.csv_bytes = pd.DataFrame({"Depth_mm": depth_array, "RF_Voltage": rf_signal, "Envelope": signal_envelope, "DAC_Curve": dac_curve}).to_csv(index=False).encode('utf-8')
            st.session_state.export_ready = True
            st.rerun() 

    if st.session_state.export_ready:
        st.success("✅ Files successfully compiled and temporarily unlocked for download!")
        st.download_button(
            label="📄 CLICK TO DOWNLOAD SECURE PDF",
            data=st.session_state.pdf_bytes,
            file_name=f"ASME_NDT_Report_{datetime.now().strftime('%H%M%S')}.pdf",
            mime="application/pdf",
            use_container_width=True
        )


st.markdown("---")
if st.session_state.export_ready:
    st.download_button("📥 Download Raw Data Matrix (.csv)", data=st.session_state.csv_bytes, file_name="NDT_Raw_Data.csv", mime="text/csv")
else:
    st.info("🔒 Raw CSV data is locked. Click 'Compile & Build PDF Report' in Tab 4 to unlock downloads.")