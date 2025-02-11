import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Title of the app
st.title("Lipid Bilayer Capacitive Test Simulation")

# Sidebar controls
st.sidebar.header("Simulation Parameters")

include_nano_pores = st.sidebar.checkbox("Include Nanopore Leak", value=True)
n_pores = st.sidebar.slider("Number of Pores", min_value=1, max_value=100, value=10)
tau_inst = st.sidebar.slider("Tau_inst (s)", min_value=0.001, max_value=0.01, step=0.001, value=0.0012)
R_single = st.sidebar.slider("R_single (Ohm)", min_value=1e8, max_value=1e10, step=1e8, value=1e9)
diameter_um = st.sidebar.slider("Diameter (μm)", min_value=50.0, max_value=1000.0, step=10.0, value=200.0)
thickness_nm = st.sidebar.slider("Thickness (nm)", min_value=2.0, max_value=10.0, step=0.5, value=5.0)

# Compute bilayer capacitance from geometry:
# C = (ε₀ * εᵣ * A) / d,
# where A = π*(diameter/2)², diameter in meters and d (thickness) in meters.
epsilon0 = 8.854e-12  # F/m, vacuum permittivity
epsilon_r = 2.1       # typical relative permittivity for lipids
diameter_m = diameter_um * 1e-6
thickness_m = thickness_nm * 1e-9
area = np.pi * (diameter_m / 2) ** 2
C_nominal = epsilon0 * epsilon_r * area / thickness_m  # in Farads

# Display computed capacitance
st.sidebar.write("Computed Bilayer Capacitance: {:.2e} F".format(C_nominal))

# Simulation parameters
n_points = 1000          # total timepoints
period = 200             # period of the voltage waveform (in timepoints)
noise_std = 5e-12        # RMS noise in capacitance (F)
dt = 1e-3                # time step in seconds (1 ms)

# Determine effective nanopore resistance (if leak is included)
if include_nano_pores:
    R_nano = R_single / n_pores  # nanopores in parallel
else:
    R_nano = np.inf  # no nanopore leak

# Generate ideal voltage waveform (triangular, in mV)
x = np.arange(n_points)
voltage = 50 * signal.sawtooth(2 * np.pi * x / period, width=0.5)
voltage[:50] = 0        # hold at 0 until t=50
voltage[-50:] = 0       # hold at 0 for the last 50 points
voltage_V = voltage * 1e-3  # convert mV to V

# RC filtering (instrumentation effect) using Euler integration:
# V_filtered[n] = V_filtered[n-1] + (dt/tau_inst) * (V_ideal[n] - V_filtered[n-1])
voltage_filtered_V = np.empty_like(voltage_V)
voltage_filtered_V[0] = voltage_V[0]
for i in range(1, n_points):
    voltage_filtered_V[i] = voltage_filtered_V[i-1] + (dt / tau_inst) * (voltage_V[i] - voltage_filtered_V[i-1])

# Compute derivative of the filtered voltage (V/s)
dVdt = np.gradient(voltage_filtered_V, dt)

# Add noise to capacitance to simulate experimental variability
noise_arr = np.random.normal(0, noise_std, size=n_points)
C_eff = C_nominal + noise_arr

# Calculate currents
I_cap = C_eff * dVdt  # capacitive current, in A
I_nano = voltage_filtered_V / R_nano if np.isfinite(R_nano) else np.zeros_like(voltage_filtered_V)  # nanopore current, in A
I_total = I_cap + I_nano  # total current, in A

# Convert currents to pA (1 A = 1e12 pA)
I_cap_pA = I_cap * 1e12
I_nano_pA = I_nano * 1e12
I_total_pA = I_total * 1e12

# Create plots using Matplotlib
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Voltage waveforms
axs[0].plot(x, voltage, label="Ideal Voltage", linestyle="--", color="blue")
axs[0].plot(x, voltage_filtered_V * 1e3, label="RC Filtered Voltage", color="orange")
axs[0].set_xlabel("Time (a.u.)")
axs[0].set_ylabel("Voltage (mV)")
axs[0].set_title("Voltage Applied")
axs[0].legend()

# Plot 2: Current components
axs[1].plot(x, I_cap_pA, label="Capacitive Current", linestyle="--", color="blue")
if include_nano_pores:
    axs[1].plot(x, I_nano_pA, label="Nanopore Current", linestyle=":", color="green")
axs[1].set_xlabel("Time (a.u.)")
axs[1].set_ylabel("Current (pA)")
axs[1].set_title("Current Components")
axs[1].legend()

# Plot 3: Total current
axs[2].plot(x, I_total_pA, label="Total Current", color="red")
axs[2].set_xlabel("Time (a.u.)")
axs[2].set_ylabel("Current (pA)")
axs[2].set_title("Total Current (Capacitive + Nanopore)")
axs[2].legend()

plt.tight_layout()

# Display the plot in Streamlit
st.pyplot(fig)
