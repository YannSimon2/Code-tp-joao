import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c, k as kb, m_p, eV

needle_size_px = 1200
u_needle_size_px= 30
needle_size_m = 1.20e-3
u_needle_size_m = 0.02e-3

conversion_factor = needle_size_m / needle_size_px

plasma_size_longi_px = np.array([240,344,356,390])
u_plasma_size_longi_px = np.array([40,20,10,10])

plasma_size_transv_px = np.array([268,378,536,606])
u_plasma_size_transv_px = np.array([50,30,20,20])


plasma_size_longi_m= plasma_size_longi_px * conversion_factor
u_plasma_size_longi_m = plasma_size_longi_m * np.sqrt((u_plasma_size_longi_px / plasma_size_longi_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2)

plasma_size_transv_m= plasma_size_transv_px * conversion_factor
u_plasma_size_transv_m = plasma_size_transv_m * np.sqrt((u_plasma_size_transv_px / plasma_size_transv_px) ** 2 + (u_needle_size_px / needle_size_px) ** 2 + (u_needle_size_m / needle_size_m) ** 2)

delays_m = np.array([0,1,2,3])
u_delays_m = 0.01

delays_s = delays_m / c
u_delays_s = u_delays_m / c

# Create figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Plot longitudinal size
ax1.errorbar(delays_s, plasma_size_longi_m, yerr=u_plasma_size_longi_m, xerr=u_delays_s, fmt='o', capsize=5, label='Data')
coeffs_longi, cov_longi = np.polyfit(delays_s, plasma_size_longi_m, 1, w=1/u_plasma_size_longi_m, cov=True)
u_coeffs_longi = np.sqrt(np.diag(cov_longi))
fit_longi = np.poly1d(coeffs_longi)
ax1.plot(delays_s, fit_longi(delays_s), 'r--', label=f'Fit: y=({coeffs_longi[0]:.2e}±{u_coeffs_longi[0]:.2e})x+({coeffs_longi[1]:.2e}±{u_coeffs_longi[1]:.2e})')
ax1.set_xlabel('Delay (s)')
ax1.set_ylabel('Plasma Size (m)')
ax1.set_title('Longitudinal Plasma Size')
ax1.legend()
ax1.grid(True)

# Calculate expansion speed and temperature
v_expansion = coeffs_longi[0]  # m/s
u_v_expansion = u_coeffs_longi[0]  # uncertainty in m/s
mi = m_p  # assuming hydrogen plasma
Te = (v_expansion**2 * mi) / (2 * kb)
u_Te = Te * (2 * u_v_expansion / v_expansion)  # propagated uncertainty
Te_eV = Te * kb / eV
u_Te_eV = u_Te * kb / eV

# Add text box with results to longitudinal plot
textstr = f'Expansion speed:\n$v = ({v_expansion:.2e} ± {u_v_expansion:.2e})$ m/s\n\nPlasma temperature (H$^+$):\n$T_e = ({Te_eV:.1f} ± {u_Te_eV:.1f})$ eV\n$T_e = ({Te:.2e} ± {u_Te:.2e})$ K'
props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Plot transverse size
ax2.errorbar(delays_s, plasma_size_transv_m, yerr=u_plasma_size_transv_m, xerr=u_delays_s, fmt='o', capsize=5, label='Data')
coeffs_transv, cov_transv = np.polyfit(delays_s, plasma_size_transv_m, 1, w=1/u_plasma_size_transv_m, cov=True)
u_coeffs_transv = np.sqrt(np.diag(cov_transv))
fit_transv = np.poly1d(coeffs_transv)
ax2.plot(delays_s, fit_transv(delays_s), 'r--', label=f'Fit: y=({coeffs_transv[0]:.2e}±{u_coeffs_transv[0]:.2e})x+({coeffs_transv[1]:.2e}±{u_coeffs_transv[1]:.2e})')
ax2.set_xlabel('Delay (s)')
ax2.set_ylabel('Plasma Size (m)')
ax2.set_title('Transverse Plasma Size')
ax2.legend()
ax2.grid(True)

plt.tight_layout()

# Print results to console
print(f"\n=== Plasma Expansion Analysis ===")
print(f"Longitudinal expansion speed: ({v_expansion:.2e} ± {u_v_expansion:.2e}) m/s")
print(f"Plasma temperature (assuming H+ ions):")
print(f"  Te = ({Te:.2e} ± {u_Te:.2e}) K")
print(f"  Te = ({Te_eV:.2f} ± {u_Te_eV:.2f}) eV")

plt.show()