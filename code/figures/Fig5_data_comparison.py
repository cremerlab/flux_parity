#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model 
import growth.viz 
import seaborn as sns
colors, _ = growth.viz.matplotlib_style()

# Load the data sets
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
palette = sns.cubehelix_palette(start=.5, rot=-.5, n_colors=12).as_hex()
markers = ['X', 's', 'd', 'o', 'v', '^', '<', '>', 'P', 'p', 'h', '*']

# Map colors to E coli sources
ecoli_sources = []
for g, d in mass_frac[mass_frac['organism']=='Escherichia coli'].groupby(['source']):
    ecoli_sources.append(g)
for g, d in elong_rate[elong_rate['organism']=='Escherichia coli'].groupby(['source']):
    if g not in ecoli_sources:
        ecoli_sources.append(g)

ecoli_palette = np.random.choice(palette, len(ecoli_sources), replace=False)
ecoli_color_map = {s:c for s, c in zip(ecoli_sources, ecoli_palette)}
ecoli_marker_map = {s:m for s, m in zip(ecoli_sources, markers)}


# Map colors to Yeast sources
yeast_sources = []
for g, d in mass_frac[mass_frac['organism']=='Saccharomyces cerevisiae'].groupby(['source']):
    yeast_sources.append(g)
for g, d in elong_rate[elong_rate['organism']=='Saccharomyces cerevisiae'].groupby(['source']):
    if g not in yeast_sources:
        yeast_sources.append(g)


yeast_palette = np.random.choice(palette, len(yeast_sources), replace=False)
yeast_color_map = {s:c for s, c in zip(yeast_sources, yeast_palette)}
yeast_marker_map = {s:m for s, m in zip(yeast_sources, markers)}


# Define the organism specific constants
gamma_max_ecoli = 20 * 3600 / 7459
gamma_max_yeast = 10 * 3600 / 11984
Kd_cAA_ecoli = 0.01
Kd_cAA_yeast  = 0.1
nu_max = np.linspace(0.001, 5, 300)

# Compute the theory curves

# Scenario I
const_phiRb_ecoli = 0.15 * np.ones_like(nu_max)
const_mu_ecoli = growth.model.steady_state_growth_rate(gamma_max_ecoli, const_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
const_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, const_phiRb_ecoli, nu_max, Kd_cAA_ecoli) * 7459/3600
const_phiRb_yeast  = 0.2 * np.ones_like(nu_max)
const_mu_yeast = growth.model.steady_state_growth_rate(gamma_max_yeast, const_phiRb_yeast, nu_max, Kd_cAA_yeast)
const_gamma_yeast = growth.model.steady_state_gamma(gamma_max_yeast, const_phiRb_yeast, nu_max, Kd_cAA_yeast) * 11984 / 3600

# Scenario II
cAA_phiRb_ecoli = nu_max / (nu_max + gamma_max_ecoli)
cAA_mu_ecoli = growth.model.steady_state_growth_rate(gamma_max_ecoli, cAA_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
cAA_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, cAA_phiRb_ecoli, nu_max, Kd_cAA_ecoli) * 7459/3600
cAA_phiRb_yeast = nu_max / (nu_max + gamma_max_yeast)
cAA_mu_yeast= growth.model.steady_state_growth_rate(gamma_max_yeast, cAA_phiRb_yeast, nu_max, Kd_cAA_yeast)
cAA_gamma_yeast= growth.model.steady_state_gamma(gamma_max_yeast, cAA_phiRb_yeast, nu_max, Kd_cAA_yeast) * 11984 / 3600

# Scenario III
opt_phiRb_ecoli = growth.model.phi_R_optimal_allocation(gamma_max_ecoli,  nu_max, Kd_cAA_ecoli) 
opt_mu_ecoli = growth.model.steady_state_growth_rate(gamma_max_ecoli,  opt_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
opt_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, opt_phiRb_ecoli,  nu_max, Kd_cAA_ecoli) * 7459/3600
opt_phiRb_yeast = growth.model.phi_R_optimal_allocation(gamma_max_yeast,  nu_max, Kd_cAA_yeast) 
opt_mu_yeast = growth.model.steady_state_growth_rate(gamma_max_yeast,  opt_phiRb_yeast, nu_max, Kd_cAA_yeast)
opt_gamma_yeast = growth.model.steady_state_gamma(gamma_max_yeast, opt_phiRb_yeast,  nu_max, Kd_cAA_yeast) * 11984 / 3600

#%%
# Set up the figure canvas
fig, ax = plt.subplots(2, 3, figsize=(6.5, 4))
ax[0,0].axis('off')
ax[1,0].axis('off')

# Add labels
for i in range(2):
    ax[i,1].set(ylabel='ribosomal allocation $\phi_{Rb}$',
            xlabel='growth rate $\lambda$ [hr$^{-1}$]')
    ax[i,2].set(ylabel='translation rate $\gamma$ [hr$^{-1}$]',
             xlabel='growth rate $\lambda$ [hr$^{-1}$]')


# Set ranges
ax[0, 1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5])
ax[0, 2].set(ylim=[5, 20], xlim=[-0.05, 2.5])
ax[1, 1].set(ylim=[0, 0.35], xlim=[-0.005, 0.9])

# Plot mass fraction
for g, d in mass_frac.groupby('organism'):
    if g == 'Escherichia coli':
        _ax = ax[0,1]
        color_map = ecoli_color_map
        marker_map = ecoli_marker_map
    else:
        color_map = yeast_color_map
        marker_map = yeast_marker_map
        _ax = ax[1, 1]

    for _g, _d in d.groupby(['source']): 
        if (_g=='Boehlke & Friesen, 1975') | (_g == 'Bonven & Gulløv, 1979'):
            alpha = 0.25
        else:
            alpha = 0.75 
        _ax.plot(_d['growth_rate_hr'], _d['mass_fraction'], ms=4,  marker=marker_map[_g],
                label='__nolegend__', alpha=alpha, linestyle='none',
                markeredgecolor='k', markeredgewidth=0.25, color=color_map[_g])


for g, d in elong_rate[elong_rate['organism']=='Escherichia coli'].groupby(['source']):
    ax[0, 2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=ecoli_marker_map[g],
                 ms=4,  linestyle='none',  label='__nolegend__', color=ecoli_color_map[g],
                 markeredgewidth=0.25, markeredgecolor='k', alpha=0.75)

for g, d in elong_rate[elong_rate['organism']=='Saccharomyces cerevisiae'].groupby(['source']):    
    if (g=='Boehlke & Friesen, 1975') | (g == 'Bonven & Gulløv, 1979'):
        alpha=0.25
    else:
        alpha = 0.75 
    ax[1, 2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values,marker=yeast_marker_map[g], 
                 color=yeast_color_map[g], markeredgewidth=0.25,  ms=4, alpha=alpha, label='__nolegend__', 
                 linestyle='none', markeredgecolor='k')





# Theory curves for E. coli
ax[0, 1].plot(const_mu_ecoli, const_phiRb_ecoli, '-', color=colors['primary_purple'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[0, 1].plot(cAA_mu_ecoli, cAA_phiRb_ecoli, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[0, 1].plot(opt_mu_ecoli, opt_phiRb_ecoli, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)
ax[0, 2].plot(const_mu_ecoli, const_gamma_ecoli, '-', color=colors['primary_purple'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[0, 2].plot(cAA_mu_ecoli, cAA_gamma_ecoli, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[0, 2].plot(opt_mu_ecoli, opt_gamma_ecoli, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)

# Theory curves for yeast
ax[1, 1].plot(const_mu_yeast, const_phiRb_yeast, '-', color=colors['primary_purple'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[1, 1].plot(cAA_mu_yeast, cAA_phiRb_yeast, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[1, 1].plot(opt_mu_yeast, opt_phiRb_yeast, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)
ax[1, 2].plot(const_mu_yeast, const_gamma_yeast, '-', color=colors['primary_purple'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[1, 2].plot(cAA_mu_yeast, cAA_gamma_yeast, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[1, 2].plot(opt_mu_yeast, opt_gamma_yeast, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)

# Plot the legends
for s in ecoli_sources:
    ax[0,0].plot([], [], ms=4, color=ecoli_color_map[s], markeredgecolor='k',
                label=s, linestyle='none', marker=ecoli_marker_map[s])
for s in yeast_sources:
    ax[1,0].plot([], [], ms=4, color=yeast_color_map[s], markeredgecolor='k',
                label=s, linestyle='none', marker=yeast_marker_map[s])

for i in range(2):
    ax[i, 0].plot([],  [], '-', lw=1, color=colors['primary_purple'], label='(I) constant $\phi_{Rb}$')
    ax[i, 0].plot([],  [], '-', lw=1, color=colors['primary_green'], label='(II) constant $\gamma$')
    ax[i, 0].plot([],  [], '-', lw=1, color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$')

ax[0,0].legend()
ax[1,0].legend()
plt.tight_layout()
plt.savefig('../../figures/Fig5_data_comparison_plots.pdf', bbox_inches='tight')

# %%
