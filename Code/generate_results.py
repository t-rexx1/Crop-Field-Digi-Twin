"""
Run the full Project 8 pipeline and save all figures for the README.
"""
import os, pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("../animations", exist_ok=True)

from write_parameters import default_parameters
from plantga.data_generation import generate_dataset
from plantga.simulation import simulate_heights
from plantga.genetic_algorithm import GeneticAlgorithmEstimator, normalized_mse

# ── 0. Parameters ────────────────────────────────────────────────────────────
params  = default_parameters()
dataset = generate_dataset(true_params=params['true_params'], **params['dataset'])
time    = dataset['time']

print(f"Plants: {dataset['positions'].shape[0]}  |  Time steps: {len(time)}")

# ── 1. Train / test trajectories ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

for ax, h_clean, h_obs, label, color in [
    (axes[0], dataset['h_clean'],      dataset['h_obs'],      'Train Season',         'tab:blue'),
    (axes[1], dataset['h_clean_test'], dataset['h_obs_test'], 'Held-out Test Season', 'tab:orange'),
]:
    ax.plot(time, h_clean.mean(axis=1), color=color,   lw=2.5,         label='Clean mean')
    ax.plot(time, h_obs.mean(axis=1),   color=color,   lw=2.5, ls='--', label='Observed mean')
    ax.set_title(f'Generated Trajectory: {label}', fontsize=12, fontweight='bold')
    ax.set_xlabel('Day'); ax.set_ylabel('Mean Plant Height')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

plt.tight_layout()
plt.savefig('../animations/trajectories.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved trajectories.png")

# ── 2. Forcing signals ────────────────────────────────────────────────────────
fig, ax = plt.subplots(3, 2, figsize=(12, 7), sharex=True)

for col, (W, F, S, title) in enumerate([
    (dataset['W'], dataset['F'], dataset['S'], 'Season 1 (Train)'),
    (dataset['W_test'], dataset['F_test'], dataset['S_test'], 'Season 2 (Held-out Test)'),
]):
    ax[0, col].plot(time, W, color='tab:blue');   ax[0, col].set_title(f'{title} Forcing', fontweight='bold')
    ax[0, col].set_ylabel('W(t)')
    ax[1, col].stem(time, F, linefmt='tab:green', markerfmt='go', basefmt='k-')
    ax[1, col].set_ylabel('F(t)')
    ax[2, col].plot(time, S, color='tab:orange'); ax[2, col].set_ylabel('S(t)')
    ax[2, col].set_xlabel('Day')
    for row in range(3):
        ax[row, col].grid(alpha=0.2)

plt.tight_layout()
plt.savefig('../animations/forcing_signals.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved forcing_signals.png")

# ── 3. Run GA ─────────────────────────────────────────────────────────────────
print("\nRunning GA (default parameters)…")
estimator = GeneticAlgorithmEstimator(bounds=params['bounds'], **params['ga'])
result    = estimator.run(dataset)
print(f"Best train cost : {result['best_train_cost']:.4f}")
print(f"Generations run : {result['generations_run']}")

with open('ga_results.pkl', 'wb') as f:
    pickle.dump(result, f)

# ── 4. GA convergence ─────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 4))
gens = range(1, len(result['history']['best_train']) + 1)
ax.semilogy(gens, result['history']['best_train'],  lw=2, label='Best cost')
ax.semilogy(gens, result['history']['mean_train'],  lw=1.5, alpha=0.6, label='Mean cost')
ax.set_xlabel('Generation'); ax.set_ylabel('Normalized MSE (log)')
ax.set_title('GA Convergence', fontsize=12, fontweight='bold')
ax.legend(); ax.grid(alpha=0.25)
plt.tight_layout()
plt.savefig('../animations/ga_convergence.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved ga_convergence.png")

# ── 5. Parameter recovery table (console + image) ────────────────────────────
true_p = params['true_params']
est_p  = result['best_params']
names  = result['param_names']

print(f"\n{'param':<10} {'true':>9} {'estimated':>12} {'abs err':>10} {'% err':>8}")
print('-' * 55)
rows = []
for n in names:
    t, e = true_p[n], est_p[n]
    rows.append((n, t, e, abs(e-t), 100*abs(e-t)/(abs(t)+1e-12)))
    print(f"{n:<10} {t:>9.4f} {e:>12.4f} {abs(e-t):>10.4f} {rows[-1][4]:>7.1f}%")

# Save table as figure
fig, ax = plt.subplots(figsize=(8, 2.8))
ax.axis('off')
col_labels = ['Parameter', 'True', 'Estimated', 'Abs Error', '% Error']
cell_data  = [[r[0], f'{r[1]:.4f}', f'{r[2]:.4f}', f'{r[3]:.4f}', f'{r[4]:.1f}%'] for r in rows]
tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               cellLoc='center', loc='center',
               colColours=['#2c6e49']*5)
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1.1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
    elif rows[row-1][4] < 10:
        cell.set_facecolor('#d4edda')   # green  = well recovered
    elif rows[row-1][4] < 35:
        cell.set_facecolor('#fff3cd')   # yellow = moderate
    else:
        cell.set_facecolor('#f8d7da')   # red    = poorly recovered
plt.title('Parameter Recovery Summary', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('../animations/parameter_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved parameter_table.png")

# ── 6. Predicted vs observed trajectories ────────────────────────────────────
h_pred_train = simulate_heights(est_p, dataset['initial_heights'],
                                time, dataset['W'], dataset['F'], dataset['S'],
                                dataset['positions'])
h_pred_test  = simulate_heights(est_p, dataset['initial_heights_test'],
                                time, dataset['W_test'], dataset['F_test'], dataset['S_test'],
                                dataset['positions'])

train_mse = normalized_mse(dataset['h_obs'],      h_pred_train)
test_mse  = normalized_mse(dataset['h_obs_test'], h_pred_test)
print(f"\nTrain norm-MSE : {train_mse:.4f}")
print(f"Test  norm-MSE : {test_mse:.4f}")

fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, h_obs, h_pred, label, color in [
    (axes[0], dataset['h_obs'],      h_pred_train, 'Season 1 (Train)',         'tab:blue'),
    (axes[1], dataset['h_obs_test'], h_pred_test,  'Season 2 (Held-out Test)', 'tab:orange'),
]:
    ax.plot(time, h_obs.mean(axis=1),  color=color,  lw=2.5, ls='--', label='Observed mean')
    ax.plot(time, h_pred.mean(axis=1), color='black', lw=2.5,         label='Predicted mean')
    mse_val = train_mse if 'Train' in label else test_mse
    ax.set_title(f'{label}  (norm-MSE = {mse_val:.4f})', fontsize=11, fontweight='bold')
    ax.set_xlabel('Day'); ax.set_ylabel('Mean Plant Height')
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=9); ax.grid(alpha=0.25)

plt.suptitle('GA Parameter Recovery: Predicted vs Observed', fontsize=13, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('../animations/ga_result.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved ga_result.png")

# ── 7. High-noise experiment ──────────────────────────────────────────────────
print("\nRunning GA on high-noise data (noise_sigma=0.20)…")
dataset_noisy = generate_dataset(true_params=params['true_params'],
                                 **{**params['dataset'], 'noise_sigma': 0.20})
est_noisy = GeneticAlgorithmEstimator(bounds=params['bounds'], **params['ga'])
result_noisy = est_noisy.run(dataset_noisy)
est_p_noisy  = result_noisy['best_params']

print(f"\n{'param':<10} {'true':>9} {'low noise':>12} {'high noise':>12}")
print('-' * 47)
rows_noisy = []
for n in names:
    t, e_low, e_high = true_p[n], est_p[n], est_p_noisy[n]
    rows_noisy.append((n, t, e_low, e_high))
    print(f"{n:<10} {t:>9.4f} {e_low:>12.4f} {e_high:>12.4f}")

# Save noise comparison table as figure
fig, ax = plt.subplots(figsize=(9, 2.8))
ax.axis('off')
col_labels = ['Parameter', 'True', 'Low Noise (σ=0.03)', 'High Noise (σ=0.20)']
cell_data  = [[r[0], f'{r[1]:.4f}', f'{r[2]:.4f}', f'{r[3]:.4f}'] for r in rows_noisy]
tbl = ax.table(cellText=cell_data, colLabels=col_labels,
               cellLoc='center', loc='center',
               colColours=['#2c6e49']*4)
tbl.auto_set_font_size(False); tbl.set_fontsize(11)
tbl.scale(1.1, 1.6)
for (row, col), cell in tbl.get_celld().items():
    if row == 0:
        cell.set_text_props(color='white', fontweight='bold')
plt.title('Effect of Noise on Parameter Recovery', fontsize=12, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('../animations/noise_comparison_table.png', dpi=150, bbox_inches='tight')
plt.close()
print("Saved noise_comparison_table.png")

print("\nAll figures saved to ../animations/")
