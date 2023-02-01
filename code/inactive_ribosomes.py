# %%
import numpy as np
import pandas as pd
import growth.model
import growth.integrate
import growth.viz
import matplotlib.pyplot as plt
import tqdm
const = growth.model.load_constants()
cor, pal = growth.viz.matplotlib_style()
markercolors = growth.viz.load_markercolors()
mass_fracs = pd.read_csv(
    '../data/main_figure_data/Fig4_ecoli_ribosomal_mass_fractions.csv')
elong_rate = pd.read_csv(
    '../data/main_figure_data/Fig4_ecoli_peptide_elongation_rates.csv')


fa = [0.75, 0.8, 0.9, 1.0]
nu_range = np.linspace(0.05, 30, 100)


fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for g, d in mass_fracs.groupby(['source']):
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
               marker=markercolors[g]['m'], markerfacecolor=markercolors[g]['c'],
               markeredgecolor=cor['primary_black'], alpha=0.5, ms=4, label='__nolegend__')

for g, d in elong_rate.groupby(['source']):
    ax[1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], linestyle='none',
               marker=markercolors[g]['m'], markerfacecolor=markercolors[g]['c'],
               markeredgecolor=cor['primary_black'], alpha=0.5, ms=4, label='__nolegend__')
ls = [':', '-.', '--', '-']
for i, f in enumerate(fa):
    lam = np.zeros_like(nu_range)
    phiRb = np.zeros_like(nu_range)
    gamma = np.zeros_like(nu_range)
    for j, nu in enumerate(tqdm.tqdm(nu_range)):
        const['nu_max'] = nu
        const['f_a'] = f
        M, M_Rb, M_Mb, TAA, TAA_star = growth.integrate.equilibrate_FPM(const)
        gamma[j] = const['gamma_max'] * TAA_star / \
            (TAA_star + const['Kd_TAA_star'])
        phiRb[j] = M_Rb / M
        lam[j] = gamma[j] * phiRb[j]
    ax[0].plot(lam, phiRb, ls[i], color=cor['primary_red'], lw=1)
    ax[1].plot(lam, gamma * const['m_Rb'] / 3600,
               ls[i], color=cor['primary_red'], lw=1, label=f'{f*100:0.0f}%')

leg = ax[1].legend(fontsize=5, title='active ribosome %')
leg.get_title().set_fontsize(6)
ax[0].set_xlim([0, 2.5])
ax[1].set_xlim([0, 2.5])
ax[0].set_ylim([0, 0.3])
ax[1].set_ylim([6, 20])
ax[0].set_xlabel('growth rate [hr$^{-1}$]')
ax[1].set_xlabel('growth rate [hr$^{-1}$]')
ax[0].set_ylabel('allocation towards ribosomes')
ax[1].set_ylabel('translation speed [AA / s]')
plt.tight_layout()
plt.savefig('../figures/FigRX_inactive_ribosomes.pdf')
