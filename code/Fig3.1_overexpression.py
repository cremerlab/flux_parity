#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import growth.model
import growth.viz
import growth.integrate
import seaborn as sns
const = growth.model.load_constants()
colors, palette = growth.viz.matplotlib_style()

# Load the datasets
scott_data = pd.read_csv('../data/main_figure_data/Fig5B_Scott2010_lacZ_overexpression.csv')
other_data = pd.read_csv('../data/main_figure_data/Fig5B_overexpression_growth_rates.csv')

# Define constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
phi_O = 0.55

# Define the metabolic rates given the scott data
nu_mapper = {}
for g, d in scott_data[scott_data['phi_X']==0].groupby(['medium', 'growth_rate_hr']):
    # Get a rough measure of what phiRb is
    phiRb = g[1] / gamma_max

    # Estimate nu
    est_nu = growth.integrate.estimate_nu_FPM(0.9 * phiRb, g[1], const, phi_O,
            verbose=True, nu_buffer=4, tol=4)
    nu_mapper[g[0]] = est_nu

#%%
# Define the range of phiX over which to compute
phiX_range = np.linspace(0, 1 - phi_O - 0.01, 100)
dt = 0.0001

# Compute the Scott case
scott_theory = pd.DataFrame([])
for medium, nu in nu_mapper.items():
    for i, phiX in enumerate(phiX_range): 
        # Equilibrate
        args = {'gamma_max':gamma_max,
                'nu_max': nu,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star':Kd_TAA_star,
                'tau': tau,
                'kappa_max':kappa_max,
                'phi_O': phi_O + phiX}
        out = growth.integrate.equilibrate_FPM(args, t_return=2, dt=dt, tol=2, max_iter=10)
        gr = np.log(out[-1][0] / out[-2][0]) / dt

        # Report the data
        scott_theory = scott_theory.append({'medium': medium,
                                            'nu_max':nu,
                                            'phiX':phiX,
                                            'lam': gr},
                                            ignore_index=True)
#%%
# Compute the relative theory
relative_theory = pd.DataFrame([])
args = {'gamma_max':gamma_max,
                'nu_max': 4,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star':Kd_TAA_star,
                'tau': tau,
                'kappa_max':kappa_max,
                'phi_O': phi_O}
out = growth.integrate.equilibrate_FPM(args, t_return=2, dt=dt, tol=2, max_iter=10)
lam_0 = np.log(out[-1][0] / out[-2][0]) / dt
for i, phiX in enumerate(phiX_range):
        args = {'gamma_max':gamma_max,
                'nu_max': 4,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star':Kd_TAA_star,
                'tau': tau,
                'kappa_max':kappa_max,
                'phi_O': phi_O + phiX}
        out = growth.integrate.equilibrate_FPM(args, t_return=2, dt=dt)
        gr = np.log(out[-1][0] / out[-2][0]) / dt
        relative_theory = relative_theory.append({'phiX': phiX,
                                                  'lam_X':gr,
                                                  'lam_0':lam_0,
                                                  'lamX_lam0':gr/lam_0},
                                                  ignore_index=True)

# %%
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].axis('off')
ax[1].set_xlabel('allocation towards\n' + r'$\beta$-galactosidase', fontsize=6)
ax[1].set_ylabel('$\lambda$\ngrowth rate [hr$^{-1}$]', fontsize=6)
ax[1].set(xlim=[-0.01, 0.4], ylim=[0, 2])
ax[2].set_xlabel('allocation towards\nexcess protein', fontsize=6)
ax[2].set_ylabel('$\lambda_X / \lambda$ \nrelative growth rate', fontsize=6)
ax[2].set(ylim=[0, 1.1], xlim=[-0.01, 0.45])
ax[1].set_yticks([0, 0.5, 1, 1.5, 2])

cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=len(scott_data['medium'].unique()))
counter = 0
for g, d in scott_data.groupby(['medium']):
    ax[1].plot(d['phi_X'], d['growth_rate_hr'], 's', ms=4, markeredgecolor='k',
                markeredgewidth=0.25, alpha=0.75, color=cmap[counter])
    counter += 1

counter = 0 
for g, d in scott_theory.groupby(['medium']):
    ax[1].plot(d['phiX'], d['lam'], '--', color=cmap[counter], lw=1, zorder=1000)
    counter += 1

n_colors = len(other_data.groupby(['medium', 'source']).count())
cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=n_colors + 1)
counter = 0
for g, d in other_data.groupby(['protein']):
    if 'galactosidase' in g:
        marker = 's'
    elif 'lactamase' in g:
        marker = 'v'
    elif 'EF-Tu' == g:
        marker = 'D'
    for _g, _d in d.groupby(['medium', 'source']):
        ax[2].plot(_d['phi_X'], _d['relative_growth_rate'], linestyle='none',
                marker=marker, color=cmap[counter], markeredgecolor='k',
                markeredgewidth=0.5, alpha=0.75, ms=4,
                label=f'{g}-{_g[1]}')
        counter += 1
    

ax[2].plot(relative_theory['phiX'], relative_theory['lamX_lam0'], '--', 
            color=colors['primary_black'], lw=1, zorder=1000)
# ax[2].legend()
plt.tight_layout()
plt.savefig('../figures/main_text/plots/Fig3_overexpression.pdf', bbox_inches='tight')
s# %%

# %%
