#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model 
import growth.viz 
colors, _ = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()

# Load the data sets
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[mass_frac['organism']=='Escherichia coli']
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong_rate = elong_rate[elong_rate['organism']=='Escherichia coli']

# Define the organism specific constants
gamma_max = const['gamma_max']
Kd_cpc =  0.03 #const['Kd_cpc']
nu_max = np.linspace(0.001, 15, 300)
phi_O = 0.55

# Compute the theory curves
# Scenario I
const_phiRb = 0.25 * np.ones_like(nu_max)
const_lam = growth.model.steady_state_growth_rate(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O)
const_gamma = growth.model.steady_state_gamma(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O) * 7459/3600

# Scenario II
cpc_phiRb = growth.model.phiRb_constant_translation(gamma_max, nu_max, 10, Kd_cpc, phi_O)
cpc_lam = growth.model.steady_state_growth_rate(gamma_max, cpc_phiRb, nu_max, Kd_cpc, phi_O)
cpc_gamma = growth.model.steady_state_gamma(gamma_max, cpc_phiRb, nu_max, Kd_cpc, phi_O) * 7459/3600

# Scenario III
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu_max, Kd_cpc, phi_O) 
opt_lam = growth.model.steady_state_growth_rate(gamma_max,  opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cpc, phi_O) * 7459/3600


#%%

# Set up the figure canvas
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2.5))
ax[0].axis('off')

# Add labels
ax[1].set(ylabel='$\phi_{Rb}$\nallocation to ribosomes',
            xlabel='growth rate\n$\lambda$ [hr$^{-1}$]')
ax[2].set(ylabel='$v_{tl}$ [AA / s$^{-1}$]\ntranslation speed',

             xlabel='growth rate\n$\lambda$ [hr$^{-1}$]')

# Set ranges
ax[1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5])
ax[2].set(ylim=[5, 20], xlim=[-0.05, 2.5])

# Plot mass fraction
for g, d in mass_frac.groupby(['source']): 
    if g!= 'Wu et al., 2021':
        ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], ms=4,  marker=mapper[g]['m'],
                label='__nolegend__', alpha=0.75, linestyle='none',
                markeredgecolor='k', markeredgewidth=0.25, color=mapper[g]['c'])


for g, d in elong_rate.groupby(['source']):
    if g!='Wu et al., 2021':
        ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=mapper[g]['m'],
                 ms=4,  linestyle='none',  label='__nolegend__', color=mapper[g]['c'],
                 markeredgewidth=0.25, markeredgecolor='k', alpha=0.75)

# Theory curves for E. coli
ax[1].plot(const_lam, const_phiRb, '-', color=colors['primary_black'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[1].plot(cpc_lam, cpc_phiRb, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[1].plot(opt_lam, opt_phiRb, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)
ax[2].plot(const_lam, const_gamma, '-', color=colors['primary_black'], label='(I) constant $\phi_{Rb}$', lw=1)
ax[2].plot(cpc_lam, cpc_gamma, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1)
ax[2].plot(opt_lam, opt_gamma, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1)


for k, v in mapper.items():
    if (k != 'Skjold et al., 1973') & (k != 'Dong et al., 1996') & (k != 'Wu et al., 2021'):
        ax[0].plot([], [], ms=4, marker=v['m'], color=v['c'], markeredgecolor='k',  
                markeredgewidth=0.25, linestyle='none', label=k)
ax[0].legend()
plt.tight_layout()
# plt.savefig('../../figures/Fig4_data_comparison_plots.pdf', bbox_inches='tight')

# %%
