import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import k as , c


energy = np.array([0.6,1.02,1.5,1.97,2.95,3.65,4.62,5.7,7.6,8.54,9.5,10.6])*1e-3
u_energy = 0.01*1e-3

plasma_size_longi_px = np.array([125,167,166,188,216,260,252,264,323,344,353,367])
u_plasma_size_longi_px = np.array([20])

plasma_size_transv_px = np.array([217,252,287,303,336,371,365,369,462,480,485,501])
u_plasma_size_transv_px = np.array([20])

needle_size_px = 1200
u_needle_size_px= 30
needle_size_m = 1.20e-3
u_needle_size_m = 0.02e-3

conversion_factor = needle_size_m / needle_size_px

plasma_size_longi_m= plasma_size_longi_px * conversion_factor
u_plasma_size_longi_m = plasma_size_longi_m * np.sqrt((u_plasma_size_longi_px / plasma_size_longi_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2) 
plasma_size_transv_m= plasma_size_transv_px * conversion_factor
u_plasma_size_transv_m = plasma_size_transv_m * np.sqrt((u_plasma_size_transv_px / plasma_size_transv_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2)

# Sedov-Taylor model: R ∝ E^(1/5)
def sedov_taylor_energy(E, C):
    return C * E**(1/5)

# Fit longitudinal size with Sedov-Taylor model
popt_longi, pcov_longi = curve_fit(sedov_taylor_energy, energy, plasma_size_longi_m,
                                    sigma=u_plasma_size_longi_m, absolute_sigma=True)
C_longi = popt_longi[0]
u_C_longi = np.sqrt(pcov_longi[0, 0])

# Fit transverse size with Sedov-Taylor model
popt_transv, pcov_transv = curve_fit(sedov_taylor_energy, energy, plasma_size_transv_m,
                                      sigma=u_plasma_size_transv_m, absolute_sigma=True)
C_transv = popt_transv[0]
u_C_transv = np.sqrt(pcov_transv[0, 0])

# Generate smooth curves for plotting
energy_smooth = np.linspace(energy.min(), energy.max(), 100)

# Atmospheric parameters for shock calculations
gamma = 1.4  # heat capacity ratio for air
rho0 = 1.225  # air density at STP (kg/m^3)
P0 = 101325  # atmospheric pressure (Pa)
T0 = 293  # ambient temperature (K)
m_air = 4.81e-26  # average mass of air molecule (kg)

# Time at which measurements were taken (ADJUST THIS BASED ON YOUR EXPERIMENT!)
# For laser-induced plasma, typical imaging delay is a few nanoseconds
t_measurement = 5.63/c  # 5 ns - CHANGE THIS to match your experimental delay

# Calculate shock parameters for each energy
# From Sedov-Taylor: R = C·E^(1/5) at time t
# Therefore: dR/dt = (2/5) · R/t (for Sedov-Taylor scaling)
R_longi = sedov_taylor_energy(energy, C_longi)
R_transv = sedov_taylor_energy(energy, C_transv)

# Shock velocity dR/dt
dRdt_longi = (2/5) * R_longi / t_measurement
dRdt_transv = (2/5) * R_transv / t_measurement

# Sound speed in ambient air
vs = np.sqrt(gamma * P0 / rho0)

# Mach number: M = (1/vs) * dR/dt
M_longi = dRdt_longi / vs
M_transv = dRdt_transv / vs

# Shock jump conditions (equations 1.3-1.6)
# Fluid velocity behind shock
U_bs_longi = (2 / (gamma + 1)) * dRdt_longi
U_bs_transv = (2 / (gamma + 1)) * dRdt_transv

# Density behind shock
rho_bs = ((gamma + 1) / (gamma - 1)) * rho0

# Pressure behind shock
P_bs_longi = (2 / (gamma + 1)) * rho0 * dRdt_longi**2
P_bs_transv = (2 / (gamma + 1)) * rho0 * dRdt_transv**2

# Temperature behind shock
T_bs_longi = (2 * gamma / (gamma + 1)) * ((gamma - 1)/(gamma + 1) * M_longi**2 + 1) * T0
T_bs_transv = (2 * gamma / (gamma + 1)) * ((gamma - 1)/(gamma + 1) * M_transv**2 + 1) * T0

plt.figure(figsize=(10, 5))
plt.errorbar(energy, plasma_size_longi_m, yerr=u_plasma_size_longi_m, xerr=u_energy, fmt='o', capsize=5, label='Longitudinal Size')
plt.errorbar(energy, plasma_size_transv_m, yerr=u_plasma_size_transv_m, xerr=u_energy, fmt='o', capsize=5, label='Transverse Size')

# Plot Sedov-Taylor fits
plt.plot(energy_smooth, sedov_taylor_energy(energy_smooth, C_longi), 'r--', 
         label=f'Longi S-T fit: R=C·E$^{{1/5}}$, C=({C_longi:.4e}±{u_C_longi:.4e})')
plt.plot(energy_smooth, sedov_taylor_energy(energy_smooth, C_transv), 'g--',
         label=f'Transv S-T fit: R=C·E$^{{1/5}}$, C=({C_transv:.4e}±{u_C_transv:.4e})')

plt.xlabel('Energy (J)')
plt.ylabel('Plasma Size (m)')
plt.title('Plasma Size vs Energy (Sedov-Taylor scaling: R ∝ E$^{1/5}$)')
plt.legend()
plt.grid()

# Create additional figure for shock parameters
fig2, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))

# Plot Mach number
ax1.plot(energy*1e3, M_longi, 'ro-', label='Longitudinal')
ax1.plot(energy*1e3, M_transv, 'go-', label='Transverse')
ax1.set_xlabel('Energy (mJ)')
ax1.set_ylabel('Mach Number M')
ax1.set_title(f'Shock Mach Number vs Energy (t={t_measurement*1e9:.1f} ns)')
ax1.legend()
ax1.grid(True)

# Plot shock velocity
ax2.plot(energy*1e3, dRdt_longi/1e3, 'ro-', label='Longitudinal')
ax2.plot(energy*1e3, dRdt_transv/1e3, 'go-', label='Transverse')
ax2.set_xlabel('Energy (mJ)')
ax2.set_ylabel('Shock Velocity (km/s)')
ax2.set_title('Shock Velocity vs Energy')
ax2.legend()
ax2.grid(True)

# Plot pressure behind shock
ax3.plot(energy*1e3, P_bs_longi/1e3, 'ro-', label='Longitudinal')
ax3.plot(energy*1e3, P_bs_transv/1e3, 'go-', label='Transverse')
ax3.axhline(y=P0/1e3, color='k', linestyle='--', alpha=0.5, label='P₀')
ax3.set_xlabel('Energy (mJ)')
ax3.set_ylabel('Pressure (kPa)')
ax3.set_title('Pressure Behind Shock vs Energy')
ax3.legend()
ax3.grid(True)

# Plot temperature behind shock
ax4.plot(energy*1e3, T_bs_longi, 'ro-', label='Longitudinal')
ax4.plot(energy*1e3, T_bs_transv, 'go-', label='Transverse')
ax4.axhline(y=T0, color='k', linestyle='--', alpha=0.5, label='T₀')
ax4.set_xlabel('Energy (mJ)')
ax4.set_ylabel('Temperature (K)')
ax4.set_title('Temperature Behind Shock vs Energy')
ax4.legend()
ax4.grid(True)

plt.tight_layout()

# Print results
print(f"\n=== Sedov-Taylor Energy Scaling Analysis ===")
print(f"\nMeasurement time: t = {t_measurement*1e9:.2f} ns")
print(f"\nLongitudinal size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_longi:.4e} ± {u_C_longi:.4e}) m·J^(-1/5)")
print(f"\nTransverse size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_transv:.4e} ± {u_C_transv:.4e}) m·J^(-1/5)")
print(f"\nRatio C_transv/C_longi = {C_transv/C_longi:.3f}")

print(f"\n=== Shock Parameters (calculated at E = {energy[-1]*1e3:.2f} mJ) ===")
print(f"\nAmbient conditions:")
print(f"  ρ₀ = {rho0} kg/m³")
print(f"  P₀ = {P0/1e3:.2f} kPa")
print(f"  T₀ = {T0} K")
print(f"  γ = {gamma}")
print(f"  Sound speed: vs = {vs:.2f} m/s")

print(f"\nLongitudinal direction:")
print(f"  Shock radius: R = {R_longi[-1]*1e3:.4f} mm")
print(f"  Shock velocity: dR/dt = {dRdt_longi[-1]:.2e} m/s")
print(f"  Mach number: M = {M_longi[-1]:.2f}")
print(f"  Fluid velocity behind shock: U_bs = {U_bs_longi[-1]:.2e} m/s")
print(f"  Density behind shock: ρ_bs = {rho_bs:.3f} kg/m³ ({rho_bs/rho0:.2f}×ρ₀)")
print(f"  Pressure behind shock: P_bs = {P_bs_longi[-1]/1e3:.2f} kPa ({P_bs_longi[-1]/P0:.2f}×P₀)")
print(f"  Temperature behind shock: T_bs = {T_bs_longi[-1]:.1f} K ({T_bs_longi[-1]/T0:.2f}×T₀)")

print(f"\nTransverse direction:")
print(f"  Shock radius: R = {R_transv[-1]*1e3:.4f} mm")
print(f"  Shock velocity: dR/dt = {dRdt_transv[-1]:.2e} m/s")
print(f"  Mach number: M = {M_transv[-1]:.2f}")
print(f"  Fluid velocity behind shock: U_bs = {U_bs_transv[-1]:.2e} m/s")
print(f"  Density behind shock: ρ_bs = {rho_bs:.3f} kg/m³ ({rho_bs/rho0:.2f}×ρ₀)")
print(f"  Pressure behind shock: P_bs = {P_bs_transv[-1]/1e3:.2f} kPa ({P_bs_transv[-1]/P0:.2f}×P₀)")
print(f"  Temperature behind shock: T_bs = {T_bs_transv[-1]:.1f} K ({T_bs_transv[-1]/T0:.2f}×T₀)")

plt.show()