import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.constants import k as kb, c


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
t_measurement = 5.63/c  # 5.63 m / c ≈ 18.8 ns - CHANGE THIS to match your experimental delay

# Infer measurement time from Sedov-Taylor model
# From theory: R(t) = ξ * (E/ρ₀)^(1/5) * t^(2/5)
# From fit: R = C * E^(1/5)
# Therefore: C = ξ * ρ₀^(-1/5) * t^(2/5)
# Solving for t: t = (C * ρ₀^(1/5) / ξ)^(5/2)

xi = 1.033  # Sedov-Taylor dimensionless constant for 3D spherical blast wave with γ=1.4

# Calculate inferred time from longitudinal data
t_inferred_longi = (C_longi * rho0**(1/5) / xi)**(5/2)
u_t_inferred_longi = t_inferred_longi * (5/2) * (u_C_longi / C_longi)

# Calculate inferred time from transverse data
t_inferred_transv = (C_transv * rho0**(1/5) / xi)**(5/2)
u_t_inferred_transv = t_inferred_transv * (5/2) * (u_C_transv / C_transv)

# Calculate shock parameters for each energy
# From Sedov-Taylor: R = C·E^(1/5) at time t
# Therefore: dR/dt = (2/5) · R/t (for Sedov-Taylor scaling)
R_longi = sedov_taylor_energy(energy, C_longi)
R_transv = sedov_taylor_energy(energy, C_transv)



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
plt.legend(fontsize=8)
plt.grid()

# Add text box with time inference
textstr = f'Time Inference from S-T model:\n'
textstr += f'Assumed: t = {t_measurement*1e9:.1f} ns\n'
textstr += f'Inferred (longi): t = {t_inferred_longi*1e9:.1f} ns\n'
textstr += f'Inferred (transv): t = {t_inferred_transv*1e9:.1f} ns\n'
textstr += f'Discrepancy (longi): {abs(t_inferred_longi - t_measurement)/t_measurement*100:.1f}%\n'
textstr += f'Discrepancy (transv): {abs(t_inferred_transv - t_measurement)/t_measurement*100:.1f}%'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
plt.text(0.02, 0.98, textstr, transform=plt.gca().transAxes, fontsize=8,
        verticalalignment='top', bbox=props)



# Print results
print(f"\n=== Sedov-Taylor Energy Scaling Analysis ===")
print(f"\nAssumed measurement time: t = {t_measurement*1e9:.2f} ns ({t_measurement*1e6:.2f} μs)")
print(f"\nLongitudinal size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_longi:.4e} ± {u_C_longi:.4e}) m·J^(-1/5)")
print(f"\nTransverse size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_transv:.4e} ± {u_C_transv:.4e}) m·J^(-1/5)")
print(f"\nRatio C_transv/C_longi = {C_transv/C_longi:.3f}")

print(f"\n=== Time Inference from Sedov-Taylor Model ===")
print(f"Using: C = ξ * ρ₀^(-1/5) * t^(2/5)  =>  t = (C * ρ₀^(1/5) / ξ)^(5/2)")
print(f"With: ξ = {xi}, ρ₀ = {rho0} kg/m³, γ = {gamma}")
print(f"\nInferred time from longitudinal data:")
print(f"  t_inferred = ({t_inferred_longi*1e9:.2f} ± {u_t_inferred_longi*1e9:.2f}) ns")
print(f"  t_inferred = ({t_inferred_longi*1e6:.3f} ± {u_t_inferred_longi*1e6:.3f}) μs")
print(f"\nInferred time from transverse data:")
print(f"  t_inferred = ({t_inferred_transv*1e9:.2f} ± {u_t_inferred_transv*1e9:.2f}) ns")
print(f"  t_inferred = ({t_inferred_transv*1e6:.3f} ± {u_t_inferred_transv*1e6:.3f}) μs")
print(f"\n=== Comparison with Actual Measurement Time ===")
print(f"Assumed time:     t_measured  = {t_measurement*1e9:.2f} ns ({t_measurement*1e6:.3f} μs)")
print(f"Inferred (longi): t_inferred  = {t_inferred_longi*1e9:.2f} ns ({t_inferred_longi*1e6:.3f} μs)")
print(f"Discrepancy (longi):           {abs(t_inferred_longi - t_measurement)*1e9:.2f} ns ({abs(t_inferred_longi - t_measurement)/t_measurement*100:.1f}%)")
print(f"Inferred (transv): t_inferred = {t_inferred_transv*1e9:.2f} ns ({t_inferred_transv*1e6:.3f} μs)")
print(f"Discrepancy (transv):          {abs(t_inferred_transv - t_measurement)*1e9:.2f} ns ({abs(t_inferred_transv - t_measurement)/t_measurement*100:.1f}%)")



plt.show()