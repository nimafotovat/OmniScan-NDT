import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.signal import find_peaks
from scipy.fft import fft, fftfreq
from fpdf import FPDF
from datetime import datetime

from ultrasonic_engine import UltrasonicSimulationEngine, Z_AIR
from ascan_pipeline import (
    run_ascan_pipeline,
    filler_to_impedance,
    min_peak_distance_samples,
    match_nearest_defect,
    classify_reflector,
)

PROJECT_TITLE = (
    "Development of a Python-Based Simulator for "
    "Ultrasonic Pulse-Echo A-Scan Signal"
)

CHART = dict(
    template="plotly_white",
    font=dict(family="Libertinus Serif, Times New Roman, serif", size=12),
    margin=dict(l=48, r=24, t=48, b=48),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=1, xanchor="right"),
)
COLORS = {
    "rf": "rgba(46, 90, 136, 0.45)",
    "rf_line": "#2E5A88",
    "envelope": "#A93226",
    "dac": "#566573",
    "marker": "#117A65",
    "fft": "#6C3483",
}


def pdf_to_bytes(pdf):
    raw = pdf.output(dest="S")
    return bytes(raw) if isinstance(raw, (bytes, bytearray)) else raw.encode("latin-1")


def ascii_safe(text):
    return text.encode("ascii", errors="replace").decode("ascii")


def value_at_depth(depth_array, signal, depth_mm):
    idx = int(np.argmin(np.abs(depth_array - float(depth_mm))))
    return float(signal[idx]), idx


def build_ascan_figure(depth, rf, envelope, dac, thickness_mm, flaw_depths):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=depth, y=rf, name="RF (pulse-echo)", mode="lines",
        line=dict(color=COLORS["rf_line"], width=0.8),
        fillcolor=COLORS["rf"], fill="tozeroy",
    ))
    fig.add_trace(go.Scatter(
        x=depth, y=envelope, name="Envelope |Hilbert|", mode="lines",
        line=dict(color=COLORS["envelope"], width=2),
    ))
    fig.add_trace(go.Scatter(
        x=depth, y=dac, name="Reference DAC", mode="lines",
        line=dict(color=COLORS["dac"], width=1.8, dash="dash"),
    ))
    for i, d in enumerate(flaw_depths, start=1):
        fig.add_vline(
            x=d, line_width=1, line_dash="dot", line_color=COLORS["marker"],
            annotation_text=f"Flaw {i}", annotation_position="top",
        )
    fig.add_vline(
        x=thickness_mm, line_width=1.5, line_dash="dash",
        line_color="#1B4F72", annotation_text="Back-wall",
    )
    fig.update_layout(
        **CHART,
        height=520,
        xaxis_title="Depth (mm)",
        yaxis_title="Amplitude (arb. units)",
        title="Simulated A-Scan (depth axis)",
    )
    return fig


def build_validation_table(engine, depth_array, envelope, probe_freq_hz, targets):
    rows = []
    for label, depth_mm, z_def in targets:
        depth_m = depth_mm / 1000.0
        t_us = engine.depth_to_tof(depth_m) * 1e6
        r_val = engine.reflection_coeff(z_def)
        a_theory = engine.expected_echo_amplitude(depth_m, probe_freq_hz, r_val)

        if label.startswith("Flaw"):
            mask = (depth_array > depth_mm - 4) & (depth_array < depth_mm + 4)
        else:
            mask = depth_array > depth_mm - 5
        if np.any(mask):
            idx_local = np.where(mask)[0]
            sim_idx = idx_local[np.argmax(envelope[idx_local])]
        else:
            sim_idx = int(np.argmax(envelope))

        sim_depth = depth_array[sim_idx]
        rows.append({
            "Reflector": label,
            "Input depth (mm)": depth_mm,
            "Theory ToF (µs)": round(t_us, 3),
            "R (signed)": round(r_val, 4),
            "Theory amplitude": round(a_theory, 5),
            "Measured peak depth (mm)": round(sim_depth, 2),
            "Depth error (mm)": round(sim_depth - depth_mm, 2),
        })
    return pd.DataFrame(rows)


st.set_page_config(
    page_title="Ultrasonic A-Scan Simulator",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "export_ready" not in st.session_state:
    st.session_state.export_ready = False
    st.session_state.pdf_bytes = None
    st.session_state.csv_bytes = None

st.markdown(
    """
    <style>
    .main-title { font-size: 1.55rem; font-weight: 700; color: #1B2631; margin-bottom: 0.1rem; }
    .sub-title { font-size: 0.95rem; color: #5D6D7E; margin-bottom: 1.2rem; line-height: 1.45; }
    div[data-testid="stMetric"] {
        background: #F8F9F9; border: 1px solid #D5D8DC;
        border-radius: 6px; padding: 0.65rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.markdown(f'<p class="main-title">{PROJECT_TITLE}</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">One-dimensional pulse-echo simulator: '
    "Gaussian excitation, acoustic propagation, reflection coefficient, "
    "and band-limited RF processing.</p>",
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Simulation inputs")
    st.subheader("Material")
    material = st.selectbox(
        "Specimen",
        ["Steel (Carbon)", "Aluminum (6061)", "Copper (Pure)", "PVC Plastic", "Custom"],
    )
    if material == "Custom":
        velocity = st.number_input("Velocity c (m/s)", 1000.0, 10000.0, 5900.0, 50.0)
        density = st.number_input("Density ρ (kg/m³)", 500.0, 20000.0, 7800.0, 50.0)
        base_alpha = st.number_input("Attenuation α₀ (Np/m)", 0.1, 20.0, 1.5, 0.1)
    else:
        presets = {
            "Steel (Carbon)": (5900.0, 7800.0, 1.5),
            "Aluminum (6061)": (6320.0, 2700.0, 1.2),
            "Copper (Pure)": (4700.0, 8960.0, 2.0),
            "PVC Plastic": (2395.0, 1380.0, 4.5),
        }
        velocity, density, base_alpha = presets[material]
        st.caption(f"c = {velocity} m/s, ρ = {density} kg/m³")

    thickness_mm = st.slider("Thickness (mm)", 20.0, 120.0, 70.0)

    st.subheader("Reflectors")
    depth_lo = 3.0
    depth_hi = max(thickness_mm - 3.0, depth_lo + 1.0)
    if "d1_mm" not in st.session_state:
        st.session_state.d1_mm = min(25.0, depth_hi)
    if "d2_mm" not in st.session_state:
        st.session_state.d2_mm = min(50.0, depth_hi)
    st.session_state.d1_mm = float(np.clip(st.session_state.d1_mm, depth_lo, depth_hi))
    st.session_state.d2_mm = float(np.clip(st.session_state.d2_mm, depth_lo, depth_hi))

    d1_mm = st.slider(
        "Flaw 1 depth (mm)", depth_lo, depth_hi, st.session_state.d1_mm, key="sl_d1"
    )
    st.session_state.d1_mm = d1_mm
    f1 = st.radio("Flaw 1 fill", ["Air Void (Crack)", "Water Inclusion"], key="f1")
    d2_mm = st.slider(
        "Flaw 2 depth (mm)", depth_lo, depth_hi, st.session_state.d2_mm, key="sl_d2"
    )
    st.session_state.d2_mm = d2_mm
    f2 = st.radio("Flaw 2 fill", ["Air Void (Crack)", "Water Inclusion"], key="f2")
    z1 = filler_to_impedance(f1)
    z2 = filler_to_impedance(f2)

    st.subheader("Transducer")
    f_probe_mhz = st.slider("Center frequency (MHz)", 1.0, 10.0, 5.0, 0.5)
    dac_label = st.selectbox(
        "DAC reference level",
        ["Strict (0.08)", "Standard (0.16)", "Relaxed (0.28)"],
    )
    ref_R = {"Strict (0.08)": 0.08, "Standard (0.16)": 0.16, "Relaxed (0.28)": 0.28}[dac_label]
    use_tgc = st.checkbox("Time-gain compensation (TGC)")
    tgc_slope = st.slider("TGC slope (dB/mm)", 0.0, 2.5, 0.5, disabled=not use_tgc)

    st.subheader("Processing")
    snr_db = st.slider("SNR (dB)", 10.0, 50.0, 30.0, 1.0)
    use_filter = st.checkbox("Bandpass filter (transducer bandwidth)", value=True)

try:
    fs = 100e6
    f_probe_hz = f_probe_mhz * 1e6
    engine = UltrasonicSimulationEngine(
        velocity, density, base_alpha, thickness_mm / 1000.0, 0.01, fs
    )
    defects = [(d1_mm / 1000.0, z1), (d2_mm / 1000.0, z2)]
    result = run_ascan_pipeline(
        engine, defects, f_probe_hz, 0.5, fs, snr_db, use_filter, ref_R, use_tgc, tgc_slope
    )

    depth = result["depth_array"]
    rf = result["rf_signal"]
    envelope = result["signal_envelope"]
    dac = result["dac_curve"]
    time_array = result["time_array"]
    tgc_gain = result["tgc_gain"]

    peak_dist = min_peak_distance_samples(velocity, fs)
    flaw_depths = [d1_mm, d2_mm]
    validation_targets = [
        ("Flaw 1", d1_mm, z1), ("Flaw 2", d2_mm, z2), ("Back-wall", thickness_mm, Z_AIR)
    ]
    val_df = build_validation_table(engine, depth, envelope, f_probe_hz, validation_targets)
    max_depth_err = float(np.max(np.abs(val_df["Depth error (mm)"].to_numpy())))
except Exception as err:
    st.error(f"Simulation failed: {err}")
    st.stop()

critical = []
for idx in find_peaks(envelope, height=dac, distance=peak_dist)[0]:
    d_mm = depth[idx]
    if not (3.0 < d_mm < thickness_mm - 2.0):
        continue
    match = match_nearest_defect(d_mm, defects)
    if match:
        _, z_m = match
        r_loc = engine.reflection_coeff(z_m)
        ftype = classify_reflector(engine, z_m)
    else:
        r_loc, ftype = 0.0, "Unassigned peak"
    critical.append({
        "Depth (mm)": round(d_mm, 2),
        "Envelope peak": round(envelope[idx], 4),
        "DAC limit": round(dac[idx], 4),
        "R": round(r_loc, 4),
        "RF sign": "+" if rf[idx] >= 0 else "-",
        "Class": ftype,
    })

above_dac = len(critical) > 0
params_key = (
    velocity, density, base_alpha, thickness_mm, d1_mm, d2_mm, f1, f2,
    f_probe_mhz, dac_label, use_tgc, tgc_slope, snr_db, use_filter,
)
if st.session_state.get("params_key") != params_key:
    st.session_state.export_ready = False
    st.session_state.pdf_bytes = None
    st.session_state.csv_bytes = None
    st.session_state.params_key = params_key

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Impedance Z", f"{engine.Z_part/1e6:.2f} MRayl")
c2.metric("Wavelength λ", f"{(velocity/f_probe_hz)*1e3:.2f} mm")
c3.metric("α (eff.)", f"{engine.effective_attenuation(f_probe_hz):.3f} Np/m")
c4.metric("Max depth error", f"{max_depth_err:.2f} mm")
c5.metric("Peaks above DAC", str(len(critical)))

tab_ascan, tab_dac, tab_valid, tab_report = st.tabs([
    "A-Scan display",
    "DAC sensitivity",
    "Analytical validation",
    "Report export",
])

with tab_ascan:
    left, right = st.columns([1.65, 1])
    with left:
        st.plotly_chart(
            build_ascan_figure(depth, rf, envelope, dac, thickness_mm, flaw_depths),
            width="stretch",
        )
        st.caption(
            "RF retains signed phase from R; envelope uses |Hilbert(RF)| only for peak comparison."
        )
    with right:
        st.subheader("Peak screening (envelope vs DAC)")
        if above_dac:
            st.dataframe(pd.DataFrame(critical), hide_index=True, width="stretch")
        else:
            st.success("No mid-wall envelope peaks exceed the reference DAC curve.")

with tab_dac:
    st.subheader("Reference curve sensitivity")
    if above_dac and critical:
        alt_levels = [("Standard (0.16)", 0.16), ("Relaxed (0.28)", 0.28)]
        for name, r_test in alt_levels:
            test_dac = engine.dac_curve(time_array, f_probe_hz, r_test)
            if use_tgc:
                test_dac = test_dac * tgc_gain
            ok = all(
                value_at_depth(depth, envelope, row["Depth (mm)"])[0]
                <= value_at_depth(depth, test_dac, row["Depth (mm)"])[0]
                for row in critical
            )
            st.write(f"**{name}:** {'would pass' if ok else 'still above threshold'}")
    else:
        st.info("Enable reflectors above DAC or lower the reference level to run sensitivity checks.")

with tab_valid:
    st.subheader("Validation against t = 2d/c and exponential attenuation")
    st.dataframe(val_df, hide_index=True, width="stretch")

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Near-field N", f"{engine.near_field_length(f_probe_hz)*1e3:.2f} mm")
    col_b.metric("R (air|steel)", f"{engine.reflection_coeff(Z_AIR):.4f}")
    col_c.metric("R (water|steel)", f"{engine.reflection_coeff(z2):.4f}")

    n = len(rf)
    spectrum = np.abs(fft(rf)[: n // 2])
    freq_mhz = fftfreq(n, 1 / fs)[: n // 2] / 1e6
    fig_f = go.Figure(go.Scatter(x=freq_mhz, y=spectrum, fill="tozeroy", name="|FFT(RF)|"))
    fig_f.add_vline(x=f_probe_mhz, line_dash="dash", line_color=COLORS["envelope"],
                    annotation_text=f"{f_probe_mhz} MHz")
    fig_f.update_layout(
        **CHART, height=320,
        title="Frequency spectrum of processed RF",
        xaxis_title="Frequency (MHz)", yaxis_title="Magnitude",
        xaxis=dict(range=[0, f_probe_mhz * 2.5]),
    )
    st.plotly_chart(fig_f, width="stretch")

with tab_report:
    st.subheader("Inspection summary export")

    def make_pdf():
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 14)
        pdf.cell(0, 9, ascii_safe(PROJECT_TITLE[:70]), ln=True, align="C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 7, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), ln=True, align="C")
        pdf.ln(4)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Specimen", ln=True)
        pdf.set_font("Arial", "", 10)
        for line in [
            f"Material: {material}",
            f"c = {velocity} m/s, rho = {density} kg/m3",
            f"Thickness: {thickness_mm} mm",
            f"Probe: {f_probe_mhz} MHz, DAC R_ref = {ref_R}",
        ]:
            pdf.cell(0, 6, ascii_safe(line), ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Validation (max depth error mm)", ln=True)
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 6, ascii_safe(f"{max_depth_err:.3f}"), ln=True)
        pdf.ln(3)
        pdf.set_font("Arial", "B", 11)
        pdf.cell(0, 7, "Disposition", ln=True)
        pdf.set_font("Arial", "", 10)
        txt = "REJECT (peaks above DAC)" if above_dac else "ACCEPT (no mid-wall peaks above DAC)"
        pdf.cell(0, 6, txt, ln=True)
        return pdf_to_bytes(pdf)

    if st.button("Generate PDF and CSV", type="primary", key="gen_report"):
        try:
            st.session_state.pdf_bytes = make_pdf()
            st.session_state.csv_bytes = pd.DataFrame({
                "depth_mm": depth,
                "rf": rf,
                "envelope": envelope,
                "dac": dac,
            }).to_csv(index=False).encode("utf-8")
            st.session_state.export_ready = True
        except Exception as e:
            st.session_state.export_ready = False
            st.error(str(e))

    if st.session_state.export_ready and st.session_state.pdf_bytes:
        st.download_button("Download PDF", st.session_state.pdf_bytes,
                           f"Ascan_Report_{datetime.now():%Y%m%d_%H%M%S}.pdf",
                           "application/pdf", key="dl_pdf")
        st.download_button("Download CSV", st.session_state.csv_bytes,
                           f"Ascan_Data_{datetime.now():%Y%m%d_%H%M%S}.csv",
                           "text/csv", key="dl_csv")
