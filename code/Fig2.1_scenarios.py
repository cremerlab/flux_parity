#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz
import growth.model 
import seaborn as sns
consts = growth.model.load_constants()
colors, palette = growth.viz.matplotlib_style()
markercolors = growth.viz.load_markercolors()
# Load the experimental data
mass_fractions = pd.read_csv('../data/main_figure_data/ecoli_ribosomal_mass_fractions.csv')
elong_rates = pd.read_csv('../data/main_figure_data/ecoli_peptide_elongation_rates.csv')

#%%
gamma_max = consts['gamma_max']
Kd_cpc = consts['Kd_cpc']
phi_O = consts['phi_O']
phiRb_range = np.linspace(0.0001, 1 - phi_O -0.001)
nu_max = np.linspace(0.2, 12.25, 8)
cmap = sns.color_palette('mako', n_colors=len(nu_max)+1)


# For each nu max, compute the scenario solution
phiRb_strat1 = np.ones(len(nu_max)) * 0.2
phiRb_strat2 = growth.model.phiRb_constant_translation(gamma_max, nu_max, 10, Kd_cpc, phi_O)
phiRb_strat3 = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
lam_strat1 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat1, nu_max, Kd_cpc, phi_O)
lam_strat2 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat2, nu_max, Kd_cpc, phi_O)
lam_strat3 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat3, nu_max, Kd_cpc, phi_O)
gamma_strat1 = growth.model.steady_state_gamma(gamma_max, phiRb_strat1, nu_max, Kd_cpc, phi_O)
gamma_strat2 = growth.model.steady_state_gamma(gamma_max, phiRb_strat2, nu_max, Kd_cpc, phi_O)
gamma_strat3 = growth.model.steady_state_gamma(gamma_max, phiRb_strat3, nu_max, Kd_cpc, phi_O)


# %% Instantiate teh figure and populate with curves
fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for a in ax:
    a.set_xlabel('allocation towards ribosomes\n$\phi_{Rb}$', fontsize=6)
ax[0].set_ylabel('$\lambda$\n growth rate [hr$^{-1}$]', fontsize=6)
ax[1].set_ylabel('$\gamma(c_{pc}^*)/\gamma_{max}$\nrelative translation rate', fontsize=6)

# Plot the curves
for i, nu in enumerate(reversed(list(nu_max))):
    lam = growth.model.steady_state_growth_rate(gamma_max, phiRb_range, nu, Kd_cpc, phi_O)
    gamma = growth.model.steady_state_gamma(gamma_max, phiRb_range, nu, Kd_cpc, phi_O)
    ax[0].plot(phiRb_range, lam, '-', lw=1, color=cmap[-i-1])  
    ax[1].plot(phiRb_range, gamma / gamma_max, '-', lw=1, color=cmap[-i-1], zorder=100-i)  

# #Plot the scenario solutions. 
# for phi, [lam, gam], [m, c] in zip([phiRb_strat1, phiRb_strat2, phiRb_strat3], 
#                            [[lam_strat1, gamma_strat1], [lam_strat2, gamma_strat2],
#                            [lam_strat3, gamma_strat3]],
#                            [['v', colors['primary_black']],
#                             ['X', colors['primary_green']],
#                             ['o', colors['primary_blue']]]):
#     ax[0].plot(phi, lam,  marker=m, color=c, ms=5, zorder=1000, linewidth=0)
#     ax[1].plot(phi, gam/gamma_max, marker=m, color=c, ms=5, zorder=1000, linewidth=0)

plt.tight_layout()
plt.savefig('../figures/main_text/Fig2.1_tents.pdf', bbox_inches='tight')

# %%
# Compute the continuous solutions for the scenarios.
nu_max = np.linspace(0.01, 20, 400)

phiRb_strat1 = 0.20 * np.ones(len(nu_max))
phiRb_strat2 = growth.model.phiRb_constant_translation(gamma_max, nu_max, 10, Kd_cpc, phi_O)
phiRb_strat3 = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
lam_strat1 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat1, nu_max, Kd_cpc, phi_O)
lam_strat2 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat2, nu_max, Kd_cpc, phi_O)
lam_strat3 = growth.model.steady_state_growth_rate(gamma_max, phiRb_strat3, nu_max, Kd_cpc, phi_O)
gamma_strat1 = growth.model.steady_state_gamma(gamma_max, phiRb_strat1, nu_max, Kd_cpc, phi_O)
gamma_strat2 = growth.model.steady_state_gamma(gamma_max, phiRb_strat2, nu_max, Kd_cpc, phi_O)
gamma_strat3 = growth.model.steady_state_gamma(gamma_max, phiRb_strat3, nu_max, Kd_cpc, phi_O)

#%%
# Instantiate the figure for the data plots
fig, ax = plt.subplots(1, 2, figsize=(4,2))
for a in ax:
    a.set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[0].set_ylabel('$\phi_{Rb}$\nallocation towards ribosomes', fontsize=6)
ax[1].set_ylabel('$v_{tl}$\ntranslation speed [AA/s]', fontsize=6)
ax[0].set_ylim([0, 0.3])
ax[1].set_ylim([5, 20])
# Populate plots with data
for g, d in mass_fractions.groupby(['source']):
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], linewidth=0, marker=markercolors[g]['m'],
               color=markercolors[g]['c'], ms=3, markeredgewidth=0.25, markeredgecolor=colors['primary_black'],
               alpha=0.75)

for g, d in elong_rates.groupby(['source']):
    ax[1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], linewidth=0, marker=markercolors[g]['m'],
               color=markercolors[g]['c'], ms=3, markeredgewidth=0.25, markeredgecolor=colors['primary_black'],
               alpha=0.75)

# Plot the solutions fo the three scenarios
for phi, lam, gam, c in zip([phiRb_strat1, phiRb_strat2, phiRb_strat3],
                            [lam_strat1, lam_strat2, lam_strat3],
                            [gamma_strat1, gamma_strat2, gamma_strat3],
                            [colors['primary_black'], colors['primary_green'], 
                            colors['primary_blue']]):
    ax[0].plot(lam, phi, '-', lw=1.5, color=c)
    ax[1].plot(lam, gam * consts['m_Rb'] / 3600, '-', lw=1.5, color=c)
plt.tight_layout()
plt.savefig('../figures/main_text/Fig2.1_data_comparison.pdf', bbox_inches='tight')
# %%
