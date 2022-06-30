#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.integrate 
import growth.viz 
import tqdm
colors, palette = growth.viz.matplotlib_style()
source_colors = growth.viz.load_markercolors()
const = growth.model.load_constants()

# Load the experimental data
mass_fractions = pd.read_csv('../data/main_figure_data/ecoli_ribosomal_mass_fractions.csv')
elongation_rate = pd.read_csv('../data/main_figure_data/ecoli_peptide_elongation_rates.csv')
ppGpp_data = pd.read_csv('../data/main_figure_data/ecoli_relative_ppGpp.csv')

#%%
# Evaluate the flux parity model over the metabolic rates
nu_range = np.linspace(0.05, 30, 300)
df = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    _args = {'gamma_max':const['gamma_max'], 
             'nu_max': nu,
             'Kd_TAA_star': const['Kd_TAA_star'],
             'Kd_TAA': const['Kd_TAA'],
             'phi_O': const['phi_O'],
             'kappa_max': const['kappa_max'],
             'tau': const['tau']}
    out = growth.integrate.equilibrate_FPM(_args, max_iter=100)
    gamma = const['gamma_max'] * out[-1] / (out[-1] + const['Kd_TAA_star'])
    balance = out[-1] / out[-2]
    phiRb = (1 - const['phi_O']) * balance / (balance + const['tau'])
    lam = gamma * phiRb
    _df = pd.DataFrame([np.array([phiRb, lam, gamma])], columns=['phiRb', 'growth_rate', 'gamma'])
    _df['TAA'] = out[-2]
    _df['TAA_star'] = out[-1]
    _df['tRNA_per_ribosome'] = ((out[-2] + out[-1]) * out[0]) / (out[1] / const['m_Rb'])
    _df['ratio']= out[-1] / out[-2]
    _df['nu_max'] = nu
    df = pd.concat([df, _df])

#%%    
ref_args = {'gamma_max':const['gamma_max'], 
        'nu_max': 4,
        'Kd_TAA_star': 0.1 * const['Kd_TAA_star'],
        'Kd_TAA': 0.1 * const['Kd_TAA'],
        'phi_O': const['phi_O'],
        'kappa_max': const['kappa_max'],
        'tau': const['tau']}

out = growth.integrate.equilibrate_FPM(ref_args)
ref_ratio = out[-1]/out[-2]
df['rel_ppGpp'] = (1 + ref_ratio) / (1 + df['ratio'])

# %%
# Instantiate the figure and format the axes.
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[0].set_ylabel('$\phi_{Rb}$\n allocation towards ribosomes', fontsize=6)
ax[1].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6) 
ax[1].set_ylabel('$v_{tl}$ \ntranslation speed [AA/s]', fontsize=6)
ax[2].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[2].set_ylabel('[ppGpp] / [ppGpp]$_0$', fontsize=6)
ax[0].set_xlim([-0.1, 2.3])
ax[1].set_xlim([-0.1, 2.1])
ax[0].set_ylim([0, 0.3])
ax[1].set_ylim([5.5, 20])
ax[2].set_ylim([0.1, 15])
ax[2].set_yscale('log')
ax[2].set_xlim([0, 2.75])
ax[2].set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])
# Compute the optimal solution
nu_range = np.linspace(0.05, 20, 300)
opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'], nu_range,
                                        const['Kd_cpc'], const['phi_O'])
opt_lam = growth.model.steady_state_growth_rate(const['gamma_max'], opt_phiRb, 
                                        nu_range, const['Kd_cpc'], 
                                        const['phi_O'])
opt_gamma = growth.model.steady_state_gamma(const['gamma_max'], opt_phiRb, 
                                            nu_range, const['Kd_cpc'], 
                                            const['phi_O'])


# Plot the data
for g, d in mass_fractions.groupby(['source']):    
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=5)
for g, d in elongation_rate.groupby(['source']):    
    ax[1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=5)

# Plot the theory curves. 
ax[0].plot(opt_lam, opt_phiRb, '-', lw=1.5, color=colors['primary_blue'])
ax[0].plot(df['growth_rate'], df['phiRb'], '--', lw=1.5, color=colors['primary_red'])
ax[1].plot(opt_lam, opt_gamma * const['m_Rb'] / 3600, '-', lw=1.5, color=colors['primary_blue'])
ax[1].plot(df['growth_rate'], df['gamma'] *  const['m_Rb'] / 3600, '--', lw=1.5, color=colors['primary_red'])
ax[2].vlines(1.0, 1E-3, 20, color=colors['light_black'], alpha=0.25, linewidth=3)
ax[2].plot(df['growth_rate'], df['rel_ppGpp'], '--', color=colors['primary_red'], lw=1.5, zorder=1000)

#
# ax[2].set_ylim([0, 10])
# Plot the relative ppGpp curves
for g, d in ppGpp_data.groupby(['source']): 
    ax[2].plot(d['growth_rate_hr'], d['relative_ppGpp'], marker=source_colors[g]['m'],
            markerfacecolor=source_colors[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
            markersize=5, alpha=0.75, linestyle='none')

plt.tight_layout()
plt.savefig('../figures/Fig3.1_flux_parity_data_plots.pdf')
# %%
