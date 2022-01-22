#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import growth.model 
import growth.viz
colors, palette = growth.viz.matplotlib_style()

# Load the data 
mass_fracs = pd.read_csv('../../data/supplement_data/yeast_ribosomal_mass_fractions.csv')
elong = pd.read_csv('../../data/supplement_data/yeast_peptide_elongation_rates.csv')

# Define the mapper
marker_mapper = {'Paulo et al., 2016': 's',
                 'Paulo et al., 2015': 'd',
                 'Metzl-Raz et al., 2017': '^',
                 'Xia et al., 2021': 'o',
                 'Bonven & Gulløv, 1979': 'h',
                 'Boehlke & Friesen, 1975': 'v',
                 'Lacroute, 1973': '>',
                 'Riba et al., 2019': 'X',
                 'Siwiak & Zielenkiewicz, 2010': '<',
                 'Waldron & Lacroute, 1975': 'D'}
seaborn_palette = sns.color_palette('Greys_r', n_colors=15).as_hex()
color_mapper = {k: np.random.choice(seaborn_palette) for k in marker_mapper.keys()}
# %%

# Compute the model solutions
nu_range = np.linspace(0, 10, 200)
gamma_max = 10 * 3600 / 11984
Kd_cpc = 0.05
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_range, Kd_cpc)
opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, nu_range, Kd_cpc)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb, nu_range, Kd_cpc) * 11984 / 3600
cpc_Kd = 1 / ((1/ (0.9)) - 1)
trans_phiRb = growth.model.phiRb_constant_translation(gamma_max, nu_range, cpc_Kd, Kd_cpc)
trans_lam = growth.model.steady_state_growth_rate(gamma_max, trans_phiRb, nu_range, Kd_cpc)
trans_gamma = growth.model.steady_state_gamma(gamma_max, trans_phiRb, nu_range, Kd_cpc) * 11984 / 3600



# Set up figure canvas
fig, ax = plt.subplots(1, 3, figsize=(6, 2.5), sharex=True)
ax[0].axis('off')
for a in ax[1:]:
    a.set_xlabel('$\lambda$ [hr$^{-1}$]\n growth rate')
    a.set_xlim([0, 0.7])
ax[1].set_ylim([0, 0.5])
ax[2].set_ylim([0, 12])
ax[1].set_ylabel('$\phi_{Rb}$\nallocation towards ribosomes')
ax[2].set_ylabel('$v_{tl}$ [AA / s]\ntranslation speed')

# Plot data
for g, d in mass_fracs.groupby('source'):
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'],
                marker=marker_mapper[g], markeredgecolor='k',
                markeredgewidth=0.5, markerfacecolor=color_mapper[g],
                label='g', linestyle='none', ms=4, alpha=0.75)

for g, d in elong.groupby('source'):
    ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'],
                marker=marker_mapper[g], markeredgecolor='k',
                markeredgewidth=0.5, markerfacecolor=color_mapper[g],
                label='g', linestyle='none', ms=4, alpha=0.75)


# Plot model
ax[1].plot(opt_lam, opt_phiRb, '-', lw=1, color=colors['primary_blue'])
ax[2].plot(opt_lam, opt_gamma, '-', lw=1, color=colors['primary_blue'])
ax[1].plot(trans_lam, trans_phiRb, '-', lw=1, color=colors['primary_green'])
ax[2].plot(trans_lam, trans_gamma, '-', lw=1, color=colors['primary_green'])

for k in marker_mapper.keys():
    ax[0].plot([], [], marker=marker_mapper[k], markeredgecolor='k', markeredgewidth=0.5,
                    markerfacecolor=color_mapper[k], alpha=0.75, ms=5, 
                    label = k, linestyle='none')
ax[0].plot([],[], '-', lw=2, color=colors['primary_blue'], label='scenario III: optimal allocation')
ax[0].plot([],[], '-', lw=2, color=colors['primary_green'], label='scenario II: constant translation rate')
ax[0].legend()
plt.tight_layout()
fig.text(0, 0.95, '(A)', fontsize=8, fontweight='bold')
fig.text(0.33, 0.95, '(B)', fontsize=8, fontweight='bold')
fig.text(0.66, 0.95, '(C)', fontsize=8, fontweight='bold')
plt.savefig('../../figures/FigS5_yeast_plots.pdf')

# %%
