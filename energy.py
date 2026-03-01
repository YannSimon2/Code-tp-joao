import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


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

# Print results
print(f"\n=== Sedov-Taylor Energy Scaling Analysis ===")
print(f"\nLongitudinal size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_longi:.4e} ± {u_C_longi:.4e}) m·J^(-1/5)")
print(f"\nTransverse size:")
print(f"  R = C·E^(1/5)")
print(f"  C = ({C_transv:.4e} ± {u_C_transv:.4e}) m·J^(-1/5)")
print(f"\nRatio C_transv/C_longi = {C_transv/C_longi:.3f}")

plt.show()