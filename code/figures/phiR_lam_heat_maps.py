#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import growth.model 
import growth.viz 
colors, _ = growth.viz.matplotlib_style()
palette = sns.color_palette('mako')

# Define the parameter constants 
Kd = 0.02 
gamma_max = 20 * 3600/ 7459
phiO = 0.35

# Set up and mesh the parameter ranges 
phiR = np.linspace(0, 1 - phiO +  0.001, 200)
nu_max = np.linspace(0.001, 10, 200)
phiR_mesh, nu_max_mesh = np.meshgrid(phiR, nu_max) 
phiP_mesh = 1 - phiO - phiR_mesh

# Compute the growth rate
growth_rate = growth.model.steady_state_growth_rate(gamma_max, nu_max_mesh, 
                                                    phiR_mesh, phiP_mesh, 
                                                    Kd)

discrete_df = []
for nu in nu_max[::10]:
    growth_rate_discrete = growth.model.steady_state_growth_rate(gamma_max, nu,
                                                             phiR, 1- phiO-phiR,
                                                             Kd)
    _df = pd.DataFrame([])
    _df['phiR'] = phiR
    _df['lam'] = growth_rate_discrete
    _df['nu_max'] = nu
    discrete_df.append(_df)
discrete_df = pd.concat(discrete_df, sort=False)

# %% Plot
fig, ax = plt.subplots(1, 2, figsize=(5, 2.4))

# Format the axes
ax[1].set_xlabel('$\phi_R$')
ax[1].set_ylabel(r'$\nu_{max} / \gamma_{max}$')
ax[1].grid(False)

# Set the labels
label_inds = [0, 49, 99, 149, 199]
nu_vals = [f'{nu_max[l]/gamma_max:0.1f}' for l in label_inds]
phiR_vals = [f'{phiR[l]:0.2f}' for l in label_inds]
ax[1].set_xticks(label_inds)
ax[1].set_yticks(label_inds)
ax[1].set_xticklabels(phiR_vals)
ax[1].set_yticklabels(nu_vals)

ax[0].set_xlabel('$\phi_R$')
ax[0].set_ylabel('$\lambda / \gamma_{max}$')

# Plot the heatmap
map_ax = ax[1].imshow(growth_rate / gamma_max, cmap='mako', origin='lower', vmax=0.32)

# Compute the contours
_inds = np.arange(len(phiR))
levels = np.array([0.1, 0.5, 1, 1.5, 2, 2.5]) / gamma_max
conts = ax[1].contour(_inds, _inds, growth_rate/gamma_max, levels=levels, 
                   colors='white')

# Specify the clabel formatter
def fmt(x):
    return f'{x:0.2f}'# + ' hr$^{-1}$'
manual_locs = [(165, 10), (150, 50), (125, 100), (125, 120), (125, 150), (100, 185)]
ax[1].clabel(conts, fmt=fmt, fontsize=4, manual=manual_locs)

# Plot the single curves
palette = sns.color_palette('mako', n_colors = len(discrete_df['nu_max'].unique()) + 3)
count = 0
for g, d in discrete_df.groupby('nu_max'):
    ax[0].plot(d['phiR'], d['lam']/gamma_max, '-', lw=0.74, color=palette[count],
               label=f'{g:0.1f}')
    count += 1
plt.tight_layout()
fig.text(0, 0.95, '[A]', fontsize=6, fontweight='bold')
fig.text(0.5, 0.95, '[B]', fontsize=6, fontweight='bold')
plt.savefig('../../figures/growth_rate_parameter_scan.svg', bbox_inches='tight')
# %%

