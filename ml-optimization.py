import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Matern
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from scipy.optimize import differential_evolution, minimize
from skopt import gp_minimize
from skopt.space import Real
from skopt.plots import plot_convergence, plot_objective
from skopt.utils import use_named_args
import warnings
warnings.filterwarnings('ignore')

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

# Extract features (tf, t1, t2, P1, P2) and target (G)
X = tests[:, [0, 1, 2, 3, 4]]  # tf, t1, t2, P1, P2
y = tests[:, 7]  # Gain

param_names = ['$t_f$', '$t_1$', '$t_2$', '$P_1$', '$P_2$']
param_units = ['ns', 'ns', 'ns', 'TW', 'TW']

# Define physical bounds for each parameter based on the data
bounds = np.array([
    [X[:, 0].min() - 0.5, X[:, 0].max() + 0.5],  # tf
    [X[:, 1].min() - 0.3, X[:, 1].max() + 0.3],  # t1
    [X[:, 2].min() - 0.5, X[:, 2].max() + 0.5],  # t2
    [X[:, 3].min() - 0.1, X[:, 3].max() + 0.1],  # P1
    [X[:, 4].min() - 5, X[:, 4].max() + 5]       # P2
])

print("="*80)
print("MACHINE LEARNING OPTIMIZATION FOR CHIC SIMULATION")
print("="*80)
print(f"\nDataset: {len(X)} samples, {X.shape[1]} parameters")
print(f"Current maximum gain in data: {y.max():.2f}")
print(f"Parameters at max gain: tf={X[np.argmax(y), 0]:.2f}, t1={X[np.argmax(y), 1]:.2f}, "
      f"t2={X[np.argmax(y), 2]:.2f}, P1={X[np.argmax(y), 3]:.2f}, P2={X[np.argmax(y), 4]:.2f}")
print("\n" + "="*80)

# Normalize features for better model performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================================================================
# 1. GAUSSIAN PROCESS REGRESSION (Best for small datasets + uncertainty)
# ============================================================================
print("\n1. GAUSSIAN PROCESS REGRESSION")
print("-" * 80)

# Use Matern kernel (good for non-smooth functions)
kernel = C(1.0, (1e-3, 1e3)) * Matern(length_scale=1.0, nu=2.5)
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True)
gp.fit(X_scaled, y)

# Cross-validation score
cv_scores = cross_val_score(gp, X_scaled, y, cv=min(5, len(X)), scoring='r2')
print(f"Cross-validation R² score: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

# Optimize using differential evolution (global optimizer)
def gp_objective(x_scaled):
    x_scaled = x_scaled.reshape(1, -1)
    pred, std = gp.predict(x_scaled, return_std=True)
    # Return negative (since we minimize) with uncertainty penalty
    return -pred[0]

# Scale bounds for optimization
bounds_scaled = [(scaler.mean_[i] - 3*scaler.scale_[i], 
                  scaler.mean_[i] + 3*scaler.scale_[i]) for i in range(5)]

result_gp = differential_evolution(gp_objective, bounds_scaled, seed=42, maxiter=1000, 
                                   popsize=30, atol=1e-7, tol=1e-7)
optimal_params_gp = scaler.inverse_transform(result_gp.x.reshape(1, -1))[0]
pred_gain_gp, std_gp = gp.predict(result_gp.x.reshape(1, -1), return_std=True)

print(f"\nOptimal parameters (GP):")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {optimal_params_gp[i]:.3f} {unit}")
print(f"\nPredicted maximum gain: {pred_gain_gp[0]:.2f} ± {std_gp[0]:.2f}")
print(f"Improvement over current max: {pred_gain_gp[0] - y.max():.2f} ({100*(pred_gain_gp[0]/y.max() - 1):.1f}%)")

# ============================================================================
# 2. RANDOM FOREST REGRESSOR
# ============================================================================
print("\n" + "="*80)
print("2. RANDOM FOREST REGRESSOR")
print("-" * 80)

rf = RandomForestRegressor(n_estimators=500, max_depth=10, min_samples_split=2, 
                           min_samples_leaf=1, random_state=42)
rf.fit(X, y)

cv_scores_rf = cross_val_score(rf, X, y, cv=min(5, len(X)), scoring='r2')
print(f"Cross-validation R² score: {cv_scores_rf.mean():.3f} (+/- {cv_scores_rf.std():.3f})")

# Feature importance
print("\nFeature Importance:")
for i, (name, importance) in enumerate(zip(param_names, rf.feature_importances_)):
    print(f"  {name}: {importance:.3f}")

# Optimize
def rf_objective(x):
    return -rf.predict(x.reshape(1, -1))[0]

result_rf = differential_evolution(rf_objective, bounds, seed=42, maxiter=1000)
optimal_params_rf = result_rf.x
pred_gain_rf = -result_rf.fun

print(f"\nOptimal parameters (RF):")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {optimal_params_rf[i]:.3f} {unit}")
print(f"\nPredicted maximum gain: {pred_gain_rf:.2f}")
print(f"Improvement over current max: {pred_gain_rf - y.max():.2f} ({100*(pred_gain_rf/y.max() - 1):.1f}%)")

# ============================================================================
# 3. GRADIENT BOOSTING REGRESSOR
# ============================================================================
print("\n" + "="*80)
print("3. GRADIENT BOOSTING REGRESSOR")
print("-" * 80)

gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, 
                               min_samples_split=2, random_state=42)
gb.fit(X, y)

cv_scores_gb = cross_val_score(gb, X, y, cv=min(5, len(X)), scoring='r2')
print(f"Cross-validation R² score: {cv_scores_gb.mean():.3f} (+/- {cv_scores_gb.std():.3f})")

# Optimize
def gb_objective(x):
    return -gb.predict(x.reshape(1, -1))[0]

result_gb = differential_evolution(gb_objective, bounds, seed=42, maxiter=1000)
optimal_params_gb = result_gb.x
pred_gain_gb = -result_gb.fun

print(f"\nOptimal parameters (GB):")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {optimal_params_gb[i]:.3f} {unit}")
print(f"\nPredicted maximum gain: {pred_gain_gb:.2f}")
print(f"Improvement over current max: {pred_gain_gb - y.max():.2f} ({100*(pred_gain_gb/y.max() - 1):.1f}%)")

# ============================================================================
# 4. POLYNOMIAL REGRESSION (Baseline)
# ============================================================================
print("\n" + "="*80)
print("4. POLYNOMIAL REGRESSION (degree=2)")
print("-" * 80)

poly = PolynomialFeatures(degree=2, include_bias=True)
X_poly = poly.fit_transform(X)
lr = LinearRegression()
lr.fit(X_poly, y)

cv_scores_poly = cross_val_score(lr, X_poly, y, cv=min(5, len(X)), scoring='r2')
print(f"Cross-validation R² score: {cv_scores_poly.mean():.3f} (+/- {cv_scores_poly.std():.3f})")

# Optimize
def poly_objective(x):
    x_poly = poly.transform(x.reshape(1, -1))
    return -lr.predict(x_poly)[0]

result_poly = differential_evolution(poly_objective, bounds, seed=42, maxiter=1000)
optimal_params_poly = result_poly.x
pred_gain_poly = -result_poly.fun

print(f"\nOptimal parameters (Polynomial):")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {optimal_params_poly[i]:.3f} {unit}")
print(f"\nPredicted maximum gain: {pred_gain_poly:.2f}")
print(f"Improvement over current max: {pred_gain_poly - y.max():.2f} ({100*(pred_gain_poly/y.max() - 1):.1f}%)")

# ============================================================================
# 5. BAYESIAN OPTIMIZATION (Adaptive sampling for expensive functions)
# ============================================================================
print("\n" + "="*80)
print("5. BAYESIAN OPTIMIZATION")
print("-" * 80)

print("\nBayesian Optimization intelligently explores the parameter space")
print("by balancing exploration (uncertain regions) vs exploitation (known good regions)")
print("Ideal for expensive simulations where each evaluation is costly.\n")

# Define search space for Bayesian Optimization
search_space = [
    Real(bounds[0, 0], bounds[0, 1], name='tf'),
    Real(bounds[1, 0], bounds[1, 1], name='t1'),
    Real(bounds[2, 0], bounds[2, 1], name='t2'),
    Real(bounds[3, 0], bounds[3, 1], name='P1'),
    Real(bounds[4, 0], bounds[4, 1], name='P2')
]

# Use Gaussian Process as surrogate model (already fitted on full dataset)
# Objective function: negative gain (since we minimize)
@use_named_args(search_space)
def bayesian_objective(**params):
    x = np.array([params['tf'], params['t1'], params['t2'], params['P1'], params['P2']])
    x_scaled = scaler.transform(x.reshape(1, -1))
    pred = gp.predict(x_scaled)[0]
    return -pred  # Negative because we minimize

# Run Bayesian Optimization
# n_calls: number of evaluations
# n_initial_points: random exploration before using GP
# acq_func: acquisition function ('EI' = Expected Improvement, 'gp_hedge' = adaptive)
print("Running Bayesian Optimization with Expected Improvement acquisition...")
result_bo = gp_minimize(
    bayesian_objective,
    search_space,
    n_calls=100,  # Total number of evaluations
    n_initial_points=20,  # Random exploration phase
    acq_func='EI',  # Expected Improvement
    random_state=42,
    verbose=False
)

optimal_params_bo = np.array(result_bo.x)
pred_gain_bo = -result_bo.fun

print(f"Bayesian Optimization completed: {len(result_bo.func_vals)} evaluations")
print(f"Best gain found: {pred_gain_bo:.2f}")

print(f"\nOptimal parameters (Bayesian Opt):")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {optimal_params_bo[i]:.3f} {unit}")
print(f"\nPredicted maximum gain: {pred_gain_bo:.2f}")
print(f"Improvement over current max: {pred_gain_bo - y.max():.2f} ({100*(pred_gain_bo/y.max() - 1):.1f}%)")

# Acquisition function comparison - try different strategies
print("\n" + "-" * 80)
print("Comparing different acquisition functions:")

acq_functions = ['EI', 'LCB', 'PI']  # Expected Improvement, Lower Confidence Bound, Probability of Improvement
acq_names = ['Expected Improvement', 'Lower Confidence Bound', 'Probability of Improvement']
bo_results = {}

for acq_func, acq_name in zip(acq_functions, acq_names):
    result = gp_minimize(
        bayesian_objective,
        search_space,
        n_calls=50,
        n_initial_points=10,
        acq_func=acq_func,
        random_state=42,
        verbose=False
    )
    bo_results[acq_name] = {
        'result': result,
        'gain': -result.fun,
        'params': np.array(result.x)
    }
    print(f"  {acq_name:30s}: Gain = {-result.fun:.2f}")

# Use the best from EI for main results
best_bo_name = max(bo_results.items(), key=lambda x: x[1]['gain'])[0]
print(f"\nBest acquisition function: {best_bo_name}")

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("SUMMARY AND RECOMMENDATIONS")
print("="*80)

models = ['Gaussian Process', 'Random Forest', 'Gradient Boosting', 'Polynomial', 'Bayesian Opt']
predictions = [pred_gain_gp[0], pred_gain_rf, pred_gain_gb, pred_gain_poly, pred_gain_bo]
params_list = [optimal_params_gp, optimal_params_rf, optimal_params_gb, optimal_params_poly, optimal_params_bo]

best_idx = np.argmax(predictions)
print(f"\nBest model: {models[best_idx]}")
print(f"Predicted gain: {predictions[best_idx]:.2f}")
print(f"\nRecommended parameters to test in simulation:")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {params_list[best_idx][i]:.3f} {unit}")

# Calculate consensus (average of top 2 models)
top2_indices = np.argsort(predictions)[-2:]
consensus_params = np.mean([params_list[i] for i in top2_indices], axis=0)
consensus_gain = np.mean([predictions[i] for i in top2_indices])

print(f"\n{'Consensus (average of top 2 models):':}")
for i, (name, unit) in enumerate(zip(param_names, param_units)):
    print(f"  {name} = {consensus_params[i]:.3f} {unit}")
print(f"Expected gain: {consensus_gain:.2f}")

# ============================================================================
# VISUALIZATION
# ============================================================================

# 1. Bayesian Optimization convergence and exploration
fig_bo, axes_bo = plt.subplots(2, 2, figsize=(15, 12))
fig_bo.suptitle('Bayesian Optimization Analysis', fontsize=16, fontweight='bold')

# Plot 1: Convergence plot
ax_conv = axes_bo[0, 0]
evaluations = np.arange(1, len(result_bo.func_vals) + 1)
gains_history = -np.array(result_bo.func_vals)  # Convert back to gain (positive)
best_so_far = np.maximum.accumulate(gains_history)

ax_conv.plot(evaluations, gains_history, 'b.', alpha=0.3, label='Individual evaluations')
ax_conv.plot(evaluations, best_so_far, 'r-', linewidth=2, label='Best so far')
ax_conv.axhline(y.max(), color='green', linestyle='--', linewidth=2, label='Current data max')
ax_conv.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax_conv.set_ylabel('Predicted Gain', fontsize=12, fontweight='bold')
ax_conv.set_title('Bayesian Optimization Convergence', fontsize=14, fontweight='bold')
ax_conv.legend(fontsize=10)
ax_conv.grid(True, alpha=0.3)

# Plot 2: Exploration in P1-P2 space
ax_explore = axes_bo[0, 1]
params_history = np.array(result_bo.x_iters)
p1_history = params_history[:, 3]
p2_history = params_history[:, 4]

scatter = ax_explore.scatter(p1_history, p2_history, c=gains_history, 
                            s=100, cmap='viridis', edgecolors='black', alpha=0.7)
# Mark the start and end
ax_explore.scatter(p1_history[0], p2_history[0], s=300, marker='s', 
                  color='red', edgecolors='black', linewidth=2, label='Start', zorder=5)
ax_explore.scatter(p1_history[-1], p2_history[-1], s=300, marker='*', 
                  color='gold', edgecolors='black', linewidth=2, label='Best found', zorder=5)
# Show actual data points
ax_explore.scatter(X[:, 3], X[:, 4], s=80, marker='x', color='red', 
                  linewidth=2, label='Training data', zorder=4)

ax_explore.set_xlabel('$P_1$ (TW)', fontsize=12, fontweight='bold')
ax_explore.set_ylabel('$P_2$ (TW)', fontsize=12, fontweight='bold')
ax_explore.set_title('Bayesian Opt Exploration Path', fontsize=14, fontweight='bold')
ax_explore.legend(fontsize=10)
ax_explore.grid(True, alpha=0.3)
cbar = plt.colorbar(scatter, ax=ax_explore)
cbar.set_label('Predicted Gain', fontsize=10)

# Plot 3: Acquisition function comparison
ax_acq = axes_bo[1, 0]
acq_names_short = ['EI', 'LCB', 'PI']
acq_gains = [bo_results[name]['gain'] for name in acq_names]
colors_acq = ['#2ecc71', '#e74c3c', '#3498db']

bars = ax_acq.bar(acq_names_short, acq_gains, color=colors_acq, alpha=0.7, edgecolor='black', linewidth=2)
ax_acq.axhline(y.max(), color='orange', linestyle='--', linewidth=2, label='Current max')
ax_acq.set_ylabel('Predicted Maximum Gain', fontsize=12, fontweight='bold')
ax_acq.set_title('Acquisition Function Comparison', fontsize=14, fontweight='bold')
ax_acq.legend(fontsize=10)
ax_acq.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, gain in zip(bars, acq_gains):
    height = bar.get_height()
    ax_acq.text(bar.get_x() + bar.get_width()/2., height,
                f'{gain:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

# Plot 4: Optimization progress - parameter evolution
ax_params = axes_bo[1, 1]
iterations = np.arange(len(params_history))

# Normalize parameters to [0, 1] for comparison
params_history_norm = np.zeros_like(params_history)
for i in range(5):
    pmin, pmax = bounds[i]
    params_history_norm[:, i] = (params_history[:, i] - pmin) / (pmax - pmin)

colors_params = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
for i in range(5):
    ax_params.plot(iterations, params_history_norm[:, i], 
                   label=param_names[i], linewidth=2, alpha=0.7, color=colors_params[i])

ax_params.set_xlabel('Iteration', fontsize=12, fontweight='bold')
ax_params.set_ylabel('Normalized Parameter Value', fontsize=12, fontweight='bold')
ax_params.set_title('Parameter Evolution During Optimization', fontsize=14, fontweight='bold')
ax_params.legend(fontsize=10, loc='best')
ax_params.grid(True, alpha=0.3)
ax_params.set_ylim(-0.05, 1.05)

plt.tight_layout()
plt.show()

# 2. Model comparison
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Predictions vs Actual Gain', fontsize=16, fontweight='bold')

models_dict = {
    'Gaussian Process': (gp, X_scaled, True),
    'Random Forest': (rf, X, False),
    'Gradient Boosting': (gb, X, False),
    'Polynomial': (lr, X_poly, False)
}

for idx, (name, (model, X_data, is_scaled)) in enumerate(models_dict.items()):
    ax = axes[idx // 3, idx % 3]
    
    if is_scaled:
        y_pred = model.predict(X_data)
    else:
        y_pred = model.predict(X_data)
    
    ax.scatter(y, y_pred, alpha=0.6, s=100)
    ax.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Gain', fontsize=12)
    ax.set_ylabel('Predicted Gain', fontsize=12)
    ax.set_title(name, fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add R² score
    from sklearn.metrics import r2_score
    r2 = r2_score(y, y_pred)
    ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes, 
            fontsize=11, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# 2. Parameter sensitivity analysis using GP
print("\n" + "="*80)
print("PARAMETER SENSITIVITY ANALYSIS (using Gaussian Process)")
print("="*80)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Parameter Sensitivity Analysis - Impact on Predicted Gain', 
             fontsize=16, fontweight='bold')

baseline = optimal_params_gp.copy()

for i in range(5):
    ax = axes[i // 3, i % 3]
    
    # Create a range for this parameter
    param_range = np.linspace(bounds[i, 0], bounds[i, 1], 50)
    gains = []
    stds = []
    
    for val in param_range:
        test_params = baseline.copy()
        test_params[i] = val
        test_params_scaled = scaler.transform(test_params.reshape(1, -1))
        pred, std = gp.predict(test_params_scaled, return_std=True)
        gains.append(pred[0])
        stds.append(std[0])
    
    gains = np.array(gains)
    stds = np.array(stds)
    
    # Plot
    ax.plot(param_range, gains, 'b-', lw=2, label='Predicted Gain')
    ax.fill_between(param_range, gains - 1.96*stds, gains + 1.96*stds, 
                     alpha=0.3, label='95% confidence')
    
    # Mark optimal point
    ax.axvline(optimal_params_gp[i], color='r', linestyle='--', lw=2, label='Optimal')
    
    # Mark actual data points
    param_values = X[:, i]
    ax.scatter(param_values, y, c='green', s=50, alpha=0.6, 
               label='Actual data', zorder=5)
    
    ax.set_xlabel(f'{param_names[i]} ({param_units[i]})', fontsize=12)
    ax.set_ylabel('Gain (G)', fontsize=12)
    ax.set_title(f'Sensitivity to {param_names[i]}', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

# Remove the extra subplot
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()

# 3. Comparison of optimal parameters from different models
fig, ax = plt.subplots(figsize=(12, 8))

x = np.arange(5)
width = 0.18

for i, (model_name, params) in enumerate(zip(models, params_list)):
    offset = (i - 1.5) * width
    ax.bar(x + offset, params, width, label=model_name, alpha=0.8)

# Add baseline (current best from data)
best_current = X[np.argmax(y)]
ax.plot(x, best_current, 'r*-', markersize=15, linewidth=2, 
        label='Current Best (from data)', zorder=10)

ax.set_xlabel('Parameters', fontsize=14)
ax.set_ylabel('Parameter Values', fontsize=14)
ax.set_title('Optimal Parameters Suggested by Different ML Models', 
             fontsize=16, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f'{name}\n({unit})' for name, unit in zip(param_names, param_units)], 
                    fontsize=11)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

print("\n" + "="*80)
print("Analysis complete! Check the visualizations for insights.")
print("="*80)
