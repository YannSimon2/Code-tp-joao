import numpy as np
import matplotlib.pyplot as plt

#Analysis of data from a CHIC simulation, to see how the gain G depends on the input power P1 and the output power P2.

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


plt.figure(figsize=(10, 6))

# Get unique P2 values
p2_values = np.unique(tests[:, 4])

# Plot for each P2 value with different color
for p2 in p2_values:
    mask = tests[:, 4] == p2
    plt.scatter(tests[mask, 3], tests[mask, 7], label=f'$P_2$ = {p2} TW', s=100)

plt.xlabel('$P_1$ (TW)', fontsize=14)
plt.ylabel('G', fontsize=14)
plt.title('CHIC Simulation: Gain vs Input Power', fontsize=16, fontweight='bold')
plt.legend(fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.grid(True)
plt.show()

# Create a 2D map of G vs P1 and P2
plt.figure(figsize=(10, 8))

# Create a scatter plot with color representing G
scatter = plt.scatter(tests[:, 3], tests[:, 4], c=tests[:, 7], s=200, cmap='viridis', edgecolors='black')

# Find and highlight the maximum gain
max_gain_idx = np.argmax(tests[:, 7])
max_p1 = tests[max_gain_idx, 3]
max_p2 = tests[max_gain_idx, 4]
max_gain = tests[max_gain_idx, 7]

# Highlight the maximum gain point with a red star
plt.scatter(max_p1, max_p2, s=500, marker='*', color='red', edgecolors='black', linewidths=2, zorder=5, label=f'Max Gain: {max_gain:.1f}')

# Add annotation
plt.annotate(f'Max G = {max_gain:.1f}', 
             xy=(max_p1, max_p2), 
             xytext=(10, 10), 
             textcoords='offset points',
             fontsize=12,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))

plt.xlabel('$P_1$ (TW)', fontsize=14)
plt.ylabel('$P_2$ (TW)', fontsize=14)
plt.title('CHIC Simulation: 2D Map of Gain vs $P_1$ and $P_2$', fontsize=16, fontweight='bold')
plt.colorbar(scatter, label='G')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()

# Create a 2D map of G vs t1 and t2
plt.figure(figsize=(10, 8))

# Create a scatter plot with color representing G
scatter2 = plt.scatter(tests[:, 1], tests[:, 2], c=tests[:, 7], s=200, cmap='viridis', edgecolors='black')

# Find and highlight the maximum gain
max_gain_idx = np.argmax(tests[:, 7])
max_t1 = tests[max_gain_idx, 1]
max_t2 = tests[max_gain_idx, 2]
max_gain = tests[max_gain_idx, 7]

# Highlight the maximum gain point with a red star
plt.scatter(max_t1, max_t2, s=500, marker='*', color='red', edgecolors='black', linewidths=2, zorder=5, label=f'Max Gain: {max_gain:.1f}')

# Add annotation
plt.annotate(f'Max G = {max_gain:.1f}', 
             xy=(max_t1, max_t2), 
             xytext=(10, 10), 
             textcoords='offset points',
             fontsize=12,
             fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
             arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', color='red', lw=2))

plt.xlabel('$t_1$ (ns)', fontsize=14)
plt.ylabel('$t_2$ (ns)', fontsize=14)
plt.title('CHIC Simulation: 2D Map of Gain vs $t_1$ and $t_2$', fontsize=16, fontweight='bold')
plt.colorbar(scatter2, label='G')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.show()