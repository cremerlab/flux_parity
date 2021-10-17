#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model 
import growth.viz 
colors, palette = growth.viz.matplotlib_style()

# Load the data sets
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')

# Define the organism specific constants
gamma_max_ecoli = 20 * 3600 / 7459
gamma_max_yeast = 10 * 3600 / 11984
Kd_cAA_ecoli = 0.01
Kd_cAA_yeast  = 0.1
nu_max = np.linspace(0.001, 5, 300)

# Compute the theory curves

# Scenario I
const_phiRb_ecoli = 0.15 * np.ones_like(nu_max)
const_mu_ecoli = growth.model.steady_state_mu(gamma_max_ecoli, const_phiRb_ecoli, nu_max, Kd_cAA_yeast)
const_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, const_phiRb_ecoli, nu_max, Kd_cAA_yeast)
const_phiRb_yeast  = 0.2 * np.ones_like(nu_max)
const_mu_yeast = growth.model.steady_state_mu(gamma_max_yeast, const_phiRb_yeast, nu_max, Kd_cAA_yeast)
const_gamma_yeast = growth.model.steady_state_gamma(gamma_max_yeast, const_phiRb_yeast, nu_max, Kd_cAA_yeast)

# Scenario II
cAA_phiRb_ecoli = nu_max / (nu_max + gamma_max_ecoli)
cAA_mu_ecoli = growth.model.steady_state_mu(gamma_max_ecoli, cAA_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
cAA_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, cAA_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
cAA_phiRb_yeast = nu_max / (nu_max + gamma_max_yeast)
cAA_mu_yeast= growth.model.steady_state_mu(gamma_max_yeast, cAA_phiRb_yeast, nu_max, Kd_cAA_yeast)
cAA_gamma_yeast= growth.model.steady_state_gamma(gamma_max_yeast, cAA_phiRb_yeast, nu_max, Kd_cAA_yeast)

# Scenario III
opt_phiRb_ecoli = growth.model.phi_R_optimal_allocation(gamma_max_ecoli,  nu_max, Kd_cAA_ecoli) 
opt_mu_ecoli = growth.model.steady_state_mu(gamma_max_ecoli,  opt_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
opt_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, opt_phiRb_ecoli,  nu_max, Kd_cAA_ecoli)
opt_phiRb_yeast = growth.model.phi_R_optimal_allocation(gamma_max_yeast,  nu_max, Kd_cAA_yeast) 
opt_mu_yeast = growth.model.steady_state_mu(gamma_max_yeast,  opt_phiRb_yeast, nu_max, Kd_cAA_yeast)
opt_gamma_yeast = growth.model.steady_state_gamma(gamma_max_yeast, opt_phiRb_yeast,  nu_max, Kd_cAA_yeast)


#%%
# Set up the figure canvas
fig, ax = plt.subplots(2, 3, figsize=(5.2, 4))
ax[0,0].axis('off')
ax[1,0].axis('off')

# Add labels
for i in range(2):
    ax[i,1].set(ylabel='ribosomal mass fraction $\phi_{Rb}$',
            xlabel='growth rate $\mu$ [hr$^{-1}$]')
    ax[i,2].set(ylabel='translational efficiency $\gamma$ [hr$^{-1}$]',
             xlabel='growth rate $\mu$ [hr$^{-1}$]')


# Set ranges
ax[0, 1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5])
ax[0, 2].set(ylim=[3, 10], xlim=[-0.05, 2.5])
ax[1, 1].set(ylim=[0, 0.35], xlim=[-0.005, 0.9])
ax[1, 2].set(ylim=[0.5, 3.5], xlim=[-0.005, 0.8])

# Define markers
markers = ['s', 'o', 'd', 'X', 'v', '^']

# Plot mass fraction
for g, d in mass_frac.groupby('organism'):
    if g == 'Escherichia coli':
        _ax = ax[0,1]
    else:
        _ax = ax[1, 1]
    counter = 0
    for _g, _d in d.groupby(['source']): 
        _ax.plot(_d['growth_rate_hr'], _d['mass_fraction'], ms=5, marker=markers[counter],
                label=_g, alpha=0.75, linestyle='none')
        counter += 1

# For elongation rate, do specific mapping of colors and shapes
elong_map = {'Dai et al., 2016': [colors['primary_blue'], 'o'],
             'Forchammer & Lindahl, 1971': [colors['primary_green'], 'd'],
             'Bremmer & Dennis, 2008': [colors['primary_black'], 's'],
             'Dalbow and Young 1975' : [palette[-1], '>'],
             'Young and Bremer 1976' : [palette[-2], '<']}

for g, d in elong_rate[elong_rate['organism']=='Escherichia coli'].groupby(['source']):
    ax[0, 2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values * 3600 / 7459,
                 ms=5, marker=elong_map[g][1], color=elong_map[g][0],
                 linestyle='none', alpha=0.75, label=g)

markers = ['o', 'd']
counter = 0
for g, d in elong_rate[elong_rate['organism']=='Saccharomyces cerevisiae'].groupby(['source']):    
    print(g)
    ax[1, 2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values * 3600 / 11984, markers[counter],
                 ms=5, color='white', markeredgecolor='k', alpha=0.25, label=g)
    ax[1, 2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values * 3600 / 11984, 'x',
                 ms=6, markeredgewidth=0.5, markeredgecolor='k', alpha=0.25, label='__nolegend__')

    counter += 1


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

# Plot the theory
ax[0, 1].legend()
ax[1, 1].legend()
ax[1, 2].legend()
ax[0, 2].legend()
plt.tight_layout()
plt.savefig('../../figures/Fig5_data_comparison_plots.pdf', bbox_inches='tight')

# %%
