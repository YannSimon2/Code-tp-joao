import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, k as kb, m_p, eV
from scipy.optimize import curve_fit

needle_size_px = 1200
u_needle_size_px= 30
needle_size_m = 1.20e-3
u_needle_size_m = 0.02e-3

conversion_factor = needle_size_m / needle_size_px

plasma_size_longi_px = np.array([241,318,356,390])
u_plasma_size_longi_px = np.array([40,20,10,10])

plasma_size_transv_px = np.array([268,378,536,606])/2
u_plasma_size_transv_px = np.array([50,30,20,20])


plasma_size_longi_m= plasma_size_longi_px * conversion_factor
u_plasma_size_longi_m = plasma_size_longi_m * np.sqrt((u_plasma_size_longi_px / plasma_size_longi_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2)

plasma_size_transv_m= plasma_size_transv_px * conversion_factor
u_plasma_size_transv_m = plasma_size_transv_m * np.sqrt((u_plasma_size_transv_px / plasma_size_transv_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2)

delays_m = np.array([0.83,3.23,5.63,8.03])
u_delays_m = 0.01

delays_s = delays_m / c
u_delays_s = u_delays_m / c

# Constants for Sedov-Taylor model and shock physics
rho0 = 1.225  # air density at STP (kg/m^3)
xi = 1.033  # Sedov-Taylor dimensionless constant for 3D spherical blast wave
mi = m_p  # assuming hydrogen plasma
gamma = 1.4  # adiabatic index for air
T0 = 300  # ambient temperature (K)
P0 = 101325  # ambient pressure (Pa)

# Sedov-Taylor fit function: R(t) = A * t^(2/5)
def sedov_taylor(t, A):
    return A * t**(2/5)

# Generate smooth curve for plotting
t_smooth = np.linspace(delays_s.min(), delays_s.max(), 100)

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot longitudinal size
ax1.errorbar(delays_s, plasma_size_longi_m, yerr=u_plasma_size_longi_m, xerr=u_delays_s, fmt='o', capsize=5, label='Data')

# Linear fit
coeffs_longi, cov_longi = np.polyfit(delays_s, plasma_size_longi_m, 1, w=1/u_plasma_size_longi_m, cov=True)
u_coeffs_longi = np.sqrt(np.diag(cov_longi))
fit_longi = np.poly1d(coeffs_longi)
ax1.plot(delays_s, fit_longi(delays_s), 'r--', label=f'Linear fit: y=({coeffs_longi[0]:.4e}±{u_coeffs_longi[0]:.4e})x+({coeffs_longi[1]:.4e}±{u_coeffs_longi[1]:.4e})')

# Sedov-Taylor fit for longitudinal: R(t) = A * t^(2/5)
popt_sedov_longi, pcov_sedov_longi = curve_fit(sedov_taylor, delays_s, plasma_size_longi_m, 
                                                sigma=u_plasma_size_longi_m, absolute_sigma=True)
u_popt_sedov_longi = np.sqrt(np.diag(pcov_sedov_longi))
A_sedov_longi = popt_sedov_longi[0]
u_A_sedov_longi = u_popt_sedov_longi[0]

# Calculate laser energy from longitudinal Sedov-Taylor coefficient
Es_J_longi = rho0 * (A_sedov_longi / xi)**5
u_Es_J_longi = Es_J_longi * 5 * (u_A_sedov_longi / A_sedov_longi)
Es_mJ_longi = Es_J_longi * 1e3
u_Es_mJ_longi = u_Es_J_longi * 1e3

# Calculate shock physics quantities behind the shock (longitudinal)
# Use the mean time for evaluation
t_mean = np.mean(delays_s)
# Shock velocity dR/dt from Sedov-Taylor: dR/dt = (2/5) * A * t^(-3/5)
dRdt_longi = (2/5) * A_sedov_longi * t_mean**(-3/5)
u_dRdt_longi = dRdt_longi * (u_A_sedov_longi / A_sedov_longi)

# Sound speed in ambient air
vs = np.sqrt(gamma * P0 / rho0)

# Mach number
M_longi = dRdt_longi / vs
u_M_longi = M_longi * (u_dRdt_longi / dRdt_longi)

# Fluid velocity behind shock (Eq. 1.3)
U_bs_longi = (2 / (gamma + 1)) * dRdt_longi
u_U_bs_longi = U_bs_longi * (u_dRdt_longi / dRdt_longi)

# Density behind shock (Eq. 1.4)
rho_bs_longi = ((gamma + 1) / (gamma - 1)) * rho0
# No uncertainty since it only depends on gamma and rho0

# Pressure behind shock (Eq. 1.5)
P_bs_longi = (2 / (gamma + 1)) * rho0 * dRdt_longi**2
u_P_bs_longi = P_bs_longi * 2 * (u_dRdt_longi / dRdt_longi)

# Temperature behind shock (Eq. 1.6)
T_bs_longi = (2 * gamma / (gamma + 1)) * ((gamma - 1) / (gamma + 1) * M_longi**2 + 1) * T0
u_T_bs_longi = T_bs_longi * 2 * (gamma - 1) / (gamma + 1) * M_longi**2 / ((gamma - 1) / (gamma + 1) * M_longi**2 + 1) * (u_M_longi / M_longi)

# Plot Sedov-Taylor fit
ax1.plot(t_smooth, sedov_taylor(t_smooth, A_sedov_longi), 'g-.', 
         label=f'Sedov-Taylor fit: R=A·t$^{{2/5}}$, A=({A_sedov_longi:.4e}±{u_A_sedov_longi:.4e})')

ax1.set_xlabel('Delay (s)')
ax1.set_ylabel('Plasma Size (m)')
ax1.set_title('Longitudinal Plasma Size')
ax1.legend(fontsize=8)
ax1.grid(True)

# Calculate expansion speed and temperature
v_expansion = coeffs_longi[0]  # m/s
u_v_expansion = u_coeffs_longi[0]  # uncertainty in m/s
Te = (v_expansion**2 * mi) / (2 * kb)
u_Te = Te * (2 * u_v_expansion / v_expansion)  # propagated uncertainty
Te_eV = Te * kb / eV
u_Te_eV = u_Te * kb / eV

# Add text box with results to longitudinal plot
textstr = f'Linear model:\n$v = ({v_expansion:.4e} ± {u_v_expansion:.4e})$ m/s\n$T_e = ({Te_eV:.3f} ± {u_Te_eV:.3f})$ eV\n\nSedov-Taylor 3D ($R \\propto t^{{2/5}}$):\n$A = ({A_sedov_longi:.4e} ± {u_A_sedov_longi:.4e})$ m·s$^{{-2/5}}$\n$E_0 = ({Es_mJ_longi:.2f} ± {u_Es_mJ_longi:.2f})$ mJ'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=8,
        verticalalignment='top', bbox=props)

# Plot transverse size
ax2.errorbar(delays_s, plasma_size_transv_m, yerr=u_plasma_size_transv_m, xerr=u_delays_s, fmt='o', capsize=5, label='Data')

# Linear fit
coeffs_transv, cov_transv = np.polyfit(delays_s, plasma_size_transv_m, 1, w=1/u_plasma_size_transv_m, cov=True)
u_coeffs_transv = np.sqrt(np.diag(cov_transv))
fit_transv = np.poly1d(coeffs_transv)
ax2.plot(delays_s, fit_transv(delays_s), 'r--', label=f'Linear fit: y=({coeffs_transv[0]:.4e}±{u_coeffs_transv[0]:.4e})x+({coeffs_transv[1]:.4e}±{u_coeffs_transv[1]:.4e})')

# Sedov-Taylor fit
popt_sedov, pcov_sedov = curve_fit(sedov_taylor, delays_s, plasma_size_transv_m, 
                                    sigma=u_plasma_size_transv_m, absolute_sigma=True)
u_popt_sedov = np.sqrt(np.diag(pcov_sedov))
A_sedov = popt_sedov[0]
u_A_sedov = u_popt_sedov[0]

# Calculate laser energy from transverse Sedov-Taylor coefficient
Es_J = rho0 * (A_sedov / xi)**5
u_Es_J = Es_J * 5 * (u_A_sedov / A_sedov)  # propagated uncertainty
Es_mJ = Es_J * 1e3
u_Es_mJ = u_Es_J * 1e3

# Calculate shock physics quantities behind the shock (transverse)
# Shock velocity dR/dt from Sedov-Taylor: dR/dt = (2/5) * A * t^(-3/5)
dRdt_transv = (2/5) * A_sedov * t_mean**(-3/5)
u_dRdt_transv = dRdt_transv * (u_A_sedov / A_sedov)

# Mach number
M_transv = dRdt_transv / vs
u_M_transv = M_transv * (u_dRdt_transv / dRdt_transv)

# Fluid velocity behind shock (Eq. 1.3)
U_bs_transv = (2 / (gamma + 1)) * dRdt_transv
u_U_bs_transv = U_bs_transv * (u_dRdt_transv / dRdt_transv)

# Density behind shock (Eq. 1.4)
rho_bs_transv = ((gamma + 1) / (gamma - 1)) * rho0

# Pressure behind shock (Eq. 1.5)
P_bs_transv = (2 / (gamma + 1)) * rho0 * dRdt_transv**2
u_P_bs_transv = P_bs_transv * 2 * (u_dRdt_transv / dRdt_transv)

# Temperature behind shock (Eq. 1.6)
T_bs_transv = (2 * gamma / (gamma + 1)) * ((gamma - 1) / (gamma + 1) * M_transv**2 + 1) * T0
u_T_bs_transv = T_bs_transv * 2 * (gamma - 1) / (gamma + 1) * M_transv**2 / ((gamma - 1) / (gamma + 1) * M_transv**2 + 1) * (u_M_transv / M_transv)

# Plot Sedov-Taylor fit
ax2.plot(t_smooth, sedov_taylor(t_smooth, A_sedov), 'g-.', 
         label=f'Sedov-Taylor fit: R=A·t$^{{2/5}}$, A=({A_sedov:.4e}±{u_A_sedov:.4e})')

ax2.set_xlabel('Delay (s)')
ax2.set_ylabel('Plasma Size (m)')
ax2.set_title('Transverse Plasma Size')
ax2.legend(fontsize=8)
ax2.grid(True)

# Calculate transverse expansion speed and temperature
v_expansion_transv = coeffs_transv[0]  # m/s
u_v_expansion_transv = u_coeffs_transv[0]  # uncertainty in m/s
Te_transv = (v_expansion_transv**2 * mi) / (2 * kb)
u_Te_transv = Te_transv * (2 * u_v_expansion_transv / v_expansion_transv)  # propagated uncertainty
Te_transv_eV = Te_transv * kb / eV
u_Te_transv_eV = u_Te_transv * kb / eV

# Add text box with results to transverse plot
textstr_transv = f'Linear model:\n$v = ({v_expansion_transv:.4e} ± {u_v_expansion_transv:.4e})$ m/s\n$T_e = ({Te_transv_eV:.3f} ± {u_Te_transv_eV:.3f})$ eV\n\nSedov-Taylor 3D ($R \\propto t^{{2/5}}$):\n$A = ({A_sedov:.4e} ± {u_A_sedov:.4e})$ m·s$^{{-2/5}}$\n$E_0 = ({Es_mJ:.2f} ± {u_Es_mJ:.2f})$ mJ'
props_transv = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
ax2.text(0.05, 0.95, textstr_transv, transform=ax2.transAxes, fontsize=8,
        verticalalignment='top', bbox=props_transv)

# Calculate geometric mean temperature (equivalent isotropic temperature)
Te_geom = (Te * Te_transv**2)**(1/3)
u_Te_geom = Te_geom * (1/3) * np.sqrt((u_Te/Te)**2 + 4*(u_Te_transv/Te_transv)**2)
Te_geom_eV = Te_geom * kb / eV
u_Te_geom_eV = u_Te_geom * kb / eV

# Calculate anisotropy ratio
anisotropy_ratio = Te_transv / Te
u_anisotropy_ratio = anisotropy_ratio * np.sqrt((u_Te_transv/Te_transv)**2 + (u_Te/Te)**2)

plt.tight_layout()

# Print results to console
print(f"\n=== Plasma Expansion Analysis ===")
print(f"\nLongitudinal expansion:")
print(f"  Linear model:")
print(f"    Expansion speed: ({v_expansion:.4e} ± {u_v_expansion:.4e}) m/s")
print(f"    Plasma temperature (assuming H+ ions):")
print(f"      Te = ({Te:.4e} ± {u_Te:.4e}) K")
print(f"      Te = ({Te_eV:.4f} ± {u_Te_eV:.4f}) eV")
print(f"  Sedov-Taylor model (3D spherical, R ∝ t^(2/5)):")
print(f"    Coefficient A: ({A_sedov_longi:.4e} ± {u_A_sedov_longi:.4e}) m·s^(-2/5)")
print(f"    Laser energy E₀: ({Es_mJ_longi:.2f} ± {u_Es_mJ_longi:.2f}) mJ")
print(f"  Shock physics (at t = {t_mean*1e9:.2f} ns):")
print(f"    Shock velocity dR/dt: ({dRdt_longi:.4e} ± {u_dRdt_longi:.4e}) m/s")
print(f"    Mach number M: ({M_longi:.3f} ± {u_M_longi:.3f})")
print(f"    Fluid velocity U_bs: ({U_bs_longi:.4e} ± {u_U_bs_longi:.4e}) m/s")
print(f"    Density ρ_bs: {rho_bs_longi:.3f} kg/m³")
print(f"    Pressure P_bs: ({P_bs_longi:.4e} ± {u_P_bs_longi:.4e}) Pa")
print(f"    Temperature T_bs: ({T_bs_longi:.2f} ± {u_T_bs_longi:.2f}) K")
print(f"\nTransverse expansion:")
print(f"  Linear model:")
print(f"    Expansion speed: ({v_expansion_transv:.4e} ± {u_v_expansion_transv:.4e}) m/s")
print(f"    Plasma temperature (assuming H+ ions):")
print(f"      Te = ({Te_transv:.4e} ± {u_Te_transv:.4e}) K")
print(f"      Te = ({Te_transv_eV:.4f} ± {u_Te_transv_eV:.4f}) eV")
print(f"  Sedov-Taylor model (3D spherical, R ∝ t^(2/5)):")
print(f"    Coefficient A: ({A_sedov:.4e} ± {u_A_sedov:.4e}) m·s^(-2/5)")
print(f"    Laser energy E₀: ({Es_mJ:.2f} ± {u_Es_mJ:.2f}) mJ")
print(f"  Shock physics (at t = {t_mean*1e9:.2f} ns):")
print(f"    Shock velocity dR/dt: ({dRdt_transv:.4e} ± {u_dRdt_transv:.4e}) m/s")
print(f"    Mach number M: ({M_transv:.3f} ± {u_M_transv:.3f})")
print(f"    Fluid velocity U_bs: ({U_bs_transv:.4e} ± {u_U_bs_transv:.4e}) m/s")
print(f"    Density ρ_bs: {rho_bs_transv:.3f} kg/m³")
print(f"    Pressure P_bs: ({P_bs_transv:.4e} ± {u_P_bs_transv:.4e}) Pa")
print(f"    Temperature T_bs: ({T_bs_transv:.2f} ± {u_T_bs_transv:.2f}) K")
print(f"\nSedov-Taylor energy comparison:")
print(f"  From longitudinal: ({Es_mJ_longi:.2f} ± {u_Es_mJ_longi:.2f}) mJ")
print(f"  From transverse:   ({Es_mJ:.2f} ± {u_Es_mJ:.2f}) mJ")
print(f"  (Note: ξ={xi}, ρ₀={rho0} kg/m³ for 3D spherical blast wave with γ=1.4)")
print(f"\nGeometric mean temperature (equivalent isotropic):")
print(f"  Te_geom = (Te_long * Te_transv^2)^(1/3)")
print(f"  Te_geom = ({Te_geom:.4e} ± {u_Te_geom:.4e}) K")
print(f"  Te_geom = ({Te_geom_eV:.4f} ± {u_Te_geom_eV:.4f}) eV")
print(f"\nAnisotropy ratio:")
print(f"  Te_transv / Te_long = {anisotropy_ratio:.3f} ± {u_anisotropy_ratio:.3f}")

plt.show()