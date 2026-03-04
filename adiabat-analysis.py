import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# Data from CHIC simulation
'''format : [tf (ns),t1 (ns), t2 (ns),P1 (TW),P2 (TW),EL (MJ),ETN (MJ),G]'''
tests = np.array([[10.22, 4.15, 6.8, 0.6, 42.5, 1.65E-01, 17.4393, 1.05E+02],
                  [10.22, 4.15, 6.8, 0.5, 42.5, 1.64E-01, 0.36439, 2.22E+00],
                  [10.22, 4.15, 6.8, 0.4, 42.5, 1.62E-01, 0.017334, 1.07E-01],
                  [10.22, 4.15, 6.8, 0.7, 42.5, 1.67E-01, 17.6299, 1.06E+02],
                  [10.22, 4.7, 6.9, 0.5, 42.5, 1.58E-01, 0.10372, 6.58E-01],
                  [10.22, 4.15, 6.8, 0.65, 42.5, 1.66E-01, 17.4932, 1.05E+02],
                  [10.22, 4.15, 6.8, 0.7, 40, 1.58E-01, 15.9976, 1.02E+02],
                  [10.22, 4.15, 6.8, 0.7, 41, 1.61E-01, 17.7964, 1.10E+02],
                  [10.22, 4.15, 6.8, 0.6, 20, 8.17E-02, 0.0025813, 3.16E-02],
                  [10.22, 4.15, 6.8, 0.7, 41.5, 1.63E-01, 17.7482, 1.09E+02],
                  [10.22, 4.15, 6.8, 0.7, 42, 1.65E-01, 17.6871, 1.07E+02],
                  [10.22, 4.15, 6.34, 0.7, 41, 1.77E-01, 11.8246, 6.67E+01],
                  [10.22, 4.15, 6.6, 0.7, 41, 1.68E-01, 17.3787, 1.03E+02],
                  [10.22, 4.15, 6.5, 0.7, 41, 1.72E-01, 16.7508, 9.75E+01],
                  [10.22, 4.15, 6.9, 0.7, 41, 1.58E-01, 12.6203, 8.00E+01],
                  [10.22, 4.15, 6.8, 0.75, 41, 1.62E-01, 17.5384, 1.08E+02],
                  [11, 4, 7.6, 0.6, 40, 1.61E-01, 15.4168, 9.60E+01],
                  [10.5, 4.3, 7.6, 0.34, 40, 1.74E-01, 13.18, 7.60E+01],
                  [11, 4.3, 6.69, 0.6, 40, 1.91E-01, 8.3, 4.36E+01],
                  [10.5, 4.15, 6.69, 0.6, 42.5, 1.81E-01, 17.19, 9.48E+01]
])

# Extract parameters
tf = tests[:, 0]   # Total pulse duration (ns)
t1 = tests[:, 1]   # Foot pulse end time (ns)
t2 = tests[:, 2]   # Main pulse start time (ns)
P1 = tests[:, 3]   # Foot pulse power (TW)
P2 = tests[:, 4]   # Main pulse power (TW)
EL = tests[:, 5]   # Laser energy (MJ)
ETN = tests[:, 6]  # Thermonuclear energy (MJ)
G = tests[:, 7]    # Gain

print("="*80)
print("ADIABAT ESTIMATION FROM PULSE SHAPE PARAMETERS")
print("="*80)

# ============================================================================
# ADIABAT ESTIMATION
# ============================================================================
# In ICF, the adiabat (α) is primarily set by the foot pulse characteristics
# Lower adiabat → more efficient compression, but less stable
# Higher adiabat → more stable, but less efficient
#
# Physical basis:
# - Adiabat scales with specific entropy: α ∝ T/ρ^(γ-1)
# - For shock compression: α ∝ (shock strength)^(2(γ-1))
# - Foot pulse sets initial compression: α ∝ (P1)^β
# - Longer foot at lower power → lower adiabat (more adiabatic compression)
# - Shock timing affects entropy generation
# ============================================================================

# Method 1: Simple power law based on foot pulse intensity
# α ∝ I^(2/3) where I is intensity
# Since we don't have area, we use P1 as proxy for intensity
# Typical scaling: α ∝ P1^(0.6)

alpha_method1 = P1**0.6

# Normalize to typical ICF values (α ~ 1-4)
alpha_method1 = alpha_method1 / alpha_method1.mean() * 2.0

# Method 2: Account for pulse duration (foot pulse entropy)
# Lower intensity over longer time → lower adiabat
# α ∝ (P1/t1)^0.5 * t1^0.2 = P1^0.5 * t1^-0.3
alpha_method2 = (P1**0.5) * (t1**-0.3)
alpha_method2 = alpha_method2 / alpha_method2.mean() * 2.0

# Method 3: Include shock timing effects
# Proper shock timing (t2-t1 optimal) reduces entropy generation
# Also consider P1/P2 ratio - larger contrast can affect adiabat
shock_timing = t2 - t1
contrast_ratio = P2 / P1

# Estimate relative adiabat considering all factors
# - Higher P1 → higher α
# - Longer t1 → lower α (more gradual compression)
# - Poor shock timing → higher α (entropy generation)
# - Very high contrast → slightly higher α

# Normalize shock timing (optimal around 2-3 ns based on data)
optimal_timing = 2.5
timing_factor = 1 + 0.3 * np.abs(shock_timing - optimal_timing) / optimal_timing

alpha_method3 = (P1**0.55) * (t1**-0.25) * timing_factor
alpha_method3 = alpha_method3 / alpha_method3.mean() * 2.0

# Use Method 3 as the best estimate (most physical)
alpha_estimated = alpha_method3

print("\nAdiabat Estimation Model:")
print("  α ∝ (P1)^0.55 × (t1)^-0.25 × f(shock_timing)")
print("  Where f(timing) accounts for entropy from improperly timed shocks")
print("\nNormalized to typical ICF range: α ~ 1-4\n")

# Create results table
print("-" * 80)
print(f"{'Case':<5} {'P1':<8} {'t1':<8} {'t2-t1':<8} {'P2':<8} {'α (est)':<10} {'Gain':<10}")
print(f"{'#':<5} {'(TW)':<8} {'(ns)':<8} {'(ns)':<8} {'(TW)':<8} {'(rel.)':<10} {'(G)':<10}")
print("-" * 80)

for i in range(len(tests)):
    print(f"{i+1:<5} {P1[i]:<8.2f} {t1[i]:<8.2f} {shock_timing[i]:<8.2f} "
          f"{P2[i]:<8.2f} {alpha_estimated[i]:<10.3f} {G[i]:<10.2f}")

# Find optimal adiabat cases
high_gain_mask = G > 100
optimal_alpha_range = [alpha_estimated[high_gain_mask].min(), 
                       alpha_estimated[high_gain_mask].max()]

print("-" * 80)
print(f"\nHigh-gain cases (G > 100) have adiabat in range: "
      f"α = {optimal_alpha_range[0]:.2f} - {optimal_alpha_range[1]:.2f}")
print(f"Mean optimal adiabat: α ≈ {alpha_estimated[high_gain_mask].mean():.2f}")

# ============================================================================
# CORRELATION ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("CORRELATION ANALYSIS")
print("="*80)

# Calculate correlations
from scipy.stats import pearsonr, spearmanr

# Pearson correlation (linear)
corr_alpha_gain, p_alpha = pearsonr(alpha_estimated, G)
corr_p1_gain, p_p1 = pearsonr(P1, G)
corr_t1_gain, p_t1 = pearsonr(t1, G)
corr_timing_gain, p_timing = pearsonr(shock_timing, G)

print(f"\nPearson Correlation with Gain:")
print(f"  Adiabat (α):      r = {corr_alpha_gain:+.3f} (p = {p_alpha:.4f})")
print(f"  Foot power (P1):  r = {corr_p1_gain:+.3f} (p = {p_p1:.4f})")
print(f"  Foot time (t1):   r = {corr_t1_gain:+.3f} (p = {p_t1:.4f})")
print(f"  Shock timing:     r = {corr_timing_gain:+.3f} (p = {p_timing:.4f})")

# Spearman correlation (monotonic)
scorr_alpha_gain, sp_alpha = spearmanr(alpha_estimated, G)
print(f"\nSpearman Correlation (α vs G): ρ = {scorr_alpha_gain:+.3f} (p = {sp_alpha:.4f})")

# ============================================================================
# VISUALIZATIONS
# ============================================================================

# 1. Gain vs Adiabat
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Plot 1: Gain vs Adiabat (colored by P2)
ax1 = axes[0, 0]
scatter1 = ax1.scatter(alpha_estimated, G, c=P2, s=150, cmap='viridis', 
                       edgecolors='black', linewidth=1.5, alpha=0.8)
ax1.set_xlabel('Estimated Adiabat α (relative)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Gain (G)', fontsize=13, fontweight='bold')
ax1.set_title('Gain vs Adiabat', fontsize=15, fontweight='bold')
ax1.grid(True, alpha=0.3)
cbar1 = plt.colorbar(scatter1, ax=ax1)
cbar1.set_label('$P_2$ (TW)', fontsize=11)

# Highlight high-gain region
ax1.axhspan(100, G.max()+5, alpha=0.2, color='green', label='High Gain (G>100)')
ax1.axvspan(optimal_alpha_range[0]-0.1, optimal_alpha_range[1]+0.1, 
            alpha=0.1, color='red', label=f'Optimal α range')
ax1.legend(fontsize=10)

# Add trend line
z = np.polyfit(alpha_estimated, G, 2)
p = np.poly1d(z)
alpha_smooth = np.linspace(alpha_estimated.min(), alpha_estimated.max(), 100)
ax1.plot(alpha_smooth, p(alpha_smooth), 'r--', linewidth=2, alpha=0.5, 
         label=f'Quadratic fit')

# Plot 2: P1 vs t1 colored by adiabat
ax2 = axes[0, 1]
scatter2 = ax2.scatter(P1, t1, c=alpha_estimated, s=150, cmap='coolwarm', 
                       edgecolors='black', linewidth=1.5, alpha=0.8)
ax2.set_xlabel('Foot Pulse Power $P_1$ (TW)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Foot Pulse Duration $t_1$ (ns)', fontsize=13, fontweight='bold')
ax2.set_title('Pulse Parameters vs Adiabat', fontsize=15, fontweight='bold')
ax2.grid(True, alpha=0.3)
cbar2 = plt.colorbar(scatter2, ax=ax2)
cbar2.set_label('Adiabat α', fontsize=11)

# Add annotations for high-gain cases
for i in np.where(high_gain_mask)[0]:
    ax2.annotate(f'{G[i]:.0f}', (P1[i], t1[i]), 
                fontsize=8, ha='center', va='bottom',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6))

# Plot 3: 3D view - Adiabat vs shock timing vs Gain
ax3 = axes[1, 0]
scatter3 = ax3.scatter(alpha_estimated, shock_timing, c=G, s=150, cmap='plasma',
                       edgecolors='black', linewidth=1.5, alpha=0.8)
ax3.set_xlabel('Adiabat α', fontsize=13, fontweight='bold')
ax3.set_ylabel('Shock Timing $t_2 - t_1$ (ns)', fontsize=13, fontweight='bold')
ax3.set_title('Adiabat vs Shock Timing (color = Gain)', fontsize=15, fontweight='bold')
ax3.grid(True, alpha=0.3)
cbar3 = plt.colorbar(scatter3, ax=ax3)
cbar3.set_label('Gain (G)', fontsize=11)

# Mark optimal timing
ax3.axhline(optimal_timing, color='red', linestyle='--', linewidth=2, 
            alpha=0.5, label=f'Optimal timing (~{optimal_timing} ns)')
ax3.legend(fontsize=10)

# Plot 4: Comparison of adiabat estimation methods
ax4 = axes[1, 1]
indices = np.arange(len(tests))
width = 0.25

ax4.bar(indices - width, alpha_method1, width, label='Method 1: $P_1^{0.6}$', alpha=0.7)
ax4.bar(indices, alpha_method2, width, label='Method 2: $(P_1/t_1)$ scaling', alpha=0.7)
ax4.bar(indices + width, alpha_method3, width, label='Method 3: Full model', alpha=0.7)

ax4.set_xlabel('Case Number', fontsize=13, fontweight='bold')
ax4.set_ylabel('Estimated Adiabat α (relative)', fontsize=13, fontweight='bold')
ax4.set_title('Comparison of Adiabat Estimation Methods', fontsize=15, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3, axis='y')
ax4.set_xticks(indices)
ax4.set_xticklabels([f'{i+1}' for i in indices], fontsize=8)

plt.tight_layout()
plt.show()

# ============================================================================
# Additional Analysis: Optimal parameter space
# ============================================================================

fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))

# Left: P1 vs P2 colored by adiabat, size by gain
ax_left = axes2[0]
sizes = (G / G.max()) * 500 + 50
scatter_left = ax_left.scatter(P1, P2, c=alpha_estimated, s=sizes, 
                               cmap='coolwarm', edgecolors='black', 
                               linewidth=1.5, alpha=0.7)
ax_left.set_xlabel('Foot Pulse Power $P_1$ (TW)', fontsize=13, fontweight='bold')
ax_left.set_ylabel('Main Pulse Power $P_2$ (TW)', fontsize=13, fontweight='bold')
ax_left.set_title('Power Space (color=α, size=Gain)', fontsize=15, fontweight='bold')
ax_left.grid(True, alpha=0.3)
cbar_left = plt.colorbar(scatter_left, ax=ax_left)
cbar_left.set_label('Adiabat α', fontsize=11)

# Add legend for size
for g_val in [50, 80, 110]:
    ax_left.scatter([], [], s=(g_val/G.max())*500+50, c='gray', alpha=0.6, 
                   edgecolors='black', label=f'G = {g_val}')
ax_left.legend(scatterpoints=1, frameon=True, labelspacing=2, fontsize=10, 
              loc='upper left')

# Right: Adiabat distribution for different gain regimes
ax_right = axes2[1]

low_gain = G < 50
mid_gain = (G >= 50) & (G < 100)
high_gain = G >= 100

ax_right.hist(alpha_estimated[low_gain], bins=8, alpha=0.6, label='Low Gain (G<50)', 
             color='red', edgecolor='black')
ax_right.hist(alpha_estimated[mid_gain], bins=8, alpha=0.6, label='Mid Gain (50≤G<100)', 
             color='orange', edgecolor='black')
ax_right.hist(alpha_estimated[high_gain], bins=8, alpha=0.6, label='High Gain (G≥100)', 
             color='green', edgecolor='black')

ax_right.set_xlabel('Adiabat α (relative)', fontsize=13, fontweight='bold')
ax_right.set_ylabel('Number of Cases', fontsize=13, fontweight='bold')
ax_right.set_title('Adiabat Distribution by Gain Regime', fontsize=15, fontweight='bold')
ax_right.legend(fontsize=11)
ax_right.grid(True, alpha=0.3, axis='y')

# Add statistics
mean_alpha_high = alpha_estimated[high_gain].mean()
std_alpha_high = alpha_estimated[high_gain].std()
ax_right.axvline(mean_alpha_high, color='darkgreen', linestyle='--', linewidth=2,
                label=f'High-gain mean: {mean_alpha_high:.2f}±{std_alpha_high:.2f}')
ax_right.legend(fontsize=10)

plt.tight_layout()
plt.show()

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS FOR OPTIMAL ADIABAT")
print("="*80)

print(f"\n1. TARGET ADIABAT RANGE:")
print(f"   For maximum gain (G > 100), aim for: α ≈ {mean_alpha_high:.2f} ± {std_alpha_high:.2f}")

print(f"\n2. PULSE PARAMETERS TO ACHIEVE THIS:")
best_case_idx = np.argmax(G)
print(f"   Best case #{best_case_idx+1} has α = {alpha_estimated[best_case_idx]:.2f}")
print(f"   - P1 = {P1[best_case_idx]:.2f} TW")
print(f"   - t1 = {t1[best_case_idx]:.2f} ns")
print(f"   - t2-t1 = {shock_timing[best_case_idx]:.2f} ns")
print(f"   - P2 = {P2[best_case_idx]:.2f} TW")

print(f"\n3. KEY INSIGHTS:")
if corr_alpha_gain > 0:
    print(f"   - Higher adiabat correlates with higher gain (r={corr_alpha_gain:.3f})")
    print(f"   - This suggests: stability > compression efficiency for these parameters")
else:
    print(f"   - Lower adiabat correlates with higher gain (r={corr_alpha_gain:.3f})")
    print(f"   - This suggests: compression efficiency is key")

print(f"\n4. AVOID:")
worst_cases = np.where(G < 1)[0]
if len(worst_cases) > 0:
    print(f"   Cases with very low gain have:")
    print(f"   - Mean α = {alpha_estimated[worst_cases].mean():.2f}")
    print(f"   - These likely have too low adiabat → instabilities")

print("\n" + "="*80)
print("Analysis complete!")
print("="*80)
