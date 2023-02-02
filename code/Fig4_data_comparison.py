# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import growth.model
import growth.integrate
import growth.viz
import tqdm
colors, _ = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()

# Load the data sets
mass_frac = pd.read_csv(
    '../data/main_figure_data/Fig4_ecoli_ribosomal_mass_fractions.csv')
elong_rate = pd.read_csv(
    '../data/main_figure_data/Fig4_ecoli_peptide_elongation_rates.csv')

# Define the organism specific constants
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
nu_max = np.linspace(0.0001, 20, 500)
phi_O = const['phi_O']

# Compute the theory curves
# Scenario I
const_phiRb = 0.25 * np.ones_like(nu_max)
const_lam = growth.model.steady_state_growth_rate(
    gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O)
const_gamma = growth.model.steady_state_gamma(
    gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O) * 7459/3600

# Scenario II
cpc_phiRb = growth.model.phiRb_constant_translation(
    gamma_max, nu_max, 20, Kd_cpc, phi_O)
cpc_lam = growth.model.steady_state_growth_rate(
    gamma_max, cpc_phiRb, nu_max, Kd_cpc, phi_O)
cpc_gamma = growth.model.steady_state_gamma(
    gamma_max, cpc_phiRb, nu_max, Kd_cpc, phi_O) * 7459/3600

# Scenario III
opt_phiRb = growth.model.phiRb_optimal_allocation(
    gamma_max,  nu_max, Kd_cpc, phi_O)
opt_lam = growth.model.steady_state_growth_rate(
    gamma_max,  opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(
    gamma_max, opt_phiRb,  nu_max, Kd_cpc, phi_O) * 7459/3600

# Set up the figure canvas
fig, ax = plt.subplots(1, 2, figsize=(7, 2.5), sharex=True)

# Add labels
ax[0].set(ylabel='$M_{Rb}/M$\n ribosomal mass fraction',
          xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
          title='steady-state behavior')
ax[1].set(ylabel='$v_{tl}$ [AA / s]\n translation speed',
          xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
          title='steady-state behavior')

# Set ranges
ax[0].set(ylim=[0, 0.3], xlim=[-0.05, 2.3])
ax[1].set(ylim=[5, 20], xlim=[-0.05, 2.3])

# Plot mass fraction
counter = 100
sources = []
for g, d in mass_frac.groupby(['source']):
    if g not in sources:
        sources.append(g)
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], ms=4,  marker=mapper[g]['m'],
               label='__nolegend__', alpha=0.75, linestyle='none',
               markeredgecolor='k', markeredgewidth=0.25, color=mapper[g]['c'],
               zorder=counter)
    counter += 1
for g, d in elong_rate.groupby(['source']):
    if g not in sources:
        sources.append(g)
    ax[1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=mapper[g]['m'],
               ms=4,  linestyle='none',  label='__nolegend__', color=mapper[g]['c'],
               markeredgewidth=0.25, markeredgecolor='k', alpha=0.75)

# Theory curves for E. coli
ax[0].plot(const_lam, const_phiRb, '-', color=colors['primary_black'],
           label='(I) constant $\phi_{Rb}$', lw=1.5, zorder=1000)
ax[0].plot(cpc_lam, cpc_phiRb, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1.5,
           zorder=1000)
ax[0].plot(opt_lam, opt_phiRb, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1.5,
           zorder=1000)
ax[1].plot(const_lam, const_gamma, '-', color=colors['primary_black'], label='(I) constant $\phi_{Rb}$', lw=1.5,
           zorder=1000)
ax[1].plot(cpc_lam, cpc_gamma, '-', color=colors['primary_green'], label='(II) constant $\gamma$', lw=1.5,
           zorder=1000)
ax[1].plot(opt_lam, opt_gamma, '-', color=colors['primary_blue'], label='(III) optimal $\phi_{Rb}$', lw=1.5,
           zorder=1000)

# for g in sources:
#     ax[0].plot([], [], ms=4, marker=mapper[g]['m'], color=mapper[g]['c'], markeredgecolor='k',
#                 markeredgewidth=0.25, linestyle='none', label=g)
# plt.tight_layout()
plt.savefig('../figures/main_text/plots/Fig1.2_simple_model_comparison.pdf',
            bbox_inches='tight')

# %%
# # Generate the plots for the ppGpp model
# # Define the parameters
# nu_max = np.linspace(0.01, 20, 400)
# Kd_TAA = const['Kd_TAA']
# Kd_TAA_star = const['Kd_TAA_star']
# kappa_max = const['kappa_max']
# tau = const['tau']
# phi_O = const['phi_O']

# # #############################################################################
# # Numerically compute the optimal scenario
# # #############################################################################
# dt = 0.0001
# time_range = np.arange(0, 200, dt)
# ss_df = pd.DataFrame([])
# total_tRNA = 0.0004
# T_AA = total_tRNA / 2
# T_AA_star = total_tRNA / 2
# for i, nu in enumerate(tqdm.tqdm(nu_max)):
#     # Set the intitial state
#     _opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu, Kd_cpc)
#     M0 = 1E9
#     phi_Mb = 1 -  _opt_phiRb
#     M_Rb = _opt_phiRb * M0
#     M_Mb = phi_Mb * M0
#     params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
#     args = {'gamma_max': gamma_max,
#             'nu_max':  nu,
#             'tau': tau,
#             'Kd_TAA': Kd_TAA,
#             'Kd_TAA_star':Kd_TAA_star,
#             'kappa_max': kappa_max,
#             'phi_O': phi_O}

#     # Integrate
#     _out = growth.integrate.equilibrate_FPM(args, tol=3, max_iter=100)

#     # Compute the final props
#     ratio = _out[-1] / _out[-2]
#     tRNA_abund = _out[-2] + _out[-1]
#     biomass = _out[0]
#     gamma = gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star)

#     ss_df = ss_df.append({'phi_Rb': _out[1] / _out[0],
#                           'lam': gamma * (_out[1] /_out[0]),
#                           'gamma': gamma * 7459 / 3600,
#                           'balance': ratio,
#                           'tot_tRNA_abundance': tRNA_abund,
#                           'biomass': biomass,
#                           'tRNA_per_ribosome': (tRNA_abund * biomass) / (_out[1]/7459),
#                           'nu_max': nu},
#                           ignore_index=True)
# # %%
# fig, ax = plt.subplots(2, 3, figsize=(6.5, 2.5))
# # Add labels
# ax[1].set(ylabel='$\phi_{Rb}$\nallocation to ribosomes',
#             xlabel='growth rate\n$\lambda$ [hr$^{-1}$]')
# ax[2].set(ylabel='$v_{tl}$ [AA / s]\ntranslation speed',
#              xlabel='growth rate\n$\lambda$ [hr$^{-1}$]')
# ax[0].axis('off')

# # Set ranges
# ax[1].set(ylim=[0, 0.3], xlim=[-0.05, 2])
# ax[2].set(ylim=[5, 20], xlim=[-0.05, 2])

# # Plot mass fraction
# sources = []
# for g, d in mass_frac.groupby(['source']):
#     ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], ms=4,  marker=mapper[g]['m'],
#                 label='__nolegend__', alpha=0.75, linestyle='none',
#                 markeredgecolor='k', markeredgewidth=0.25, color=mapper[g]['c'])


# for g, d in elong_rate.groupby(['source']):
#         ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=mapper[g]['m'],
#                  ms=4,  linestyle='none',  label='__nolegend__', color=mapper[g]['c'],
#                  markeredgewidth=0.25, markeredgecolor='k', alpha=0.75)


# ax[1].plot(opt_lam, opt_phiRb, '-', color=colors['primary_blue'], lw=1.5)
# ax[1].plot(ss_df['lam'], ss_df['phi_Rb'], '--', color=colors['primary_red'], zorder=1000, lw=1.5)
# ax[2].plot(opt_lam, opt_gamma, '-', color=colors['primary_blue'], lw=1.5)
# ax[2].plot(ss_df['lam'], ss_df['gamma'], '--', color=colors['primary_red'], zorder=1000, lw=1.5)
# plt.tight_layout()
# plt.savefig('../figures/main_text/plots/Fig4_flux-parity_plots.pdf', bbox_inches='tight')

# # %%
