@echo off
title OmniScan-NDT Simulator Launcher
echo ===================================================
echo Starting Ultrasonic Simulation Dashboard...
echo Please wait while the local server starts.
echo ===================================================
call venv\Scripts\activate
streamlit run dashboard.py
pause