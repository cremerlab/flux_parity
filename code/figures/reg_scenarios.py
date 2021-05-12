#%%

import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
colors, palette = growth.viz.matplotlib_style()

#%%

# Load the relevant data 
# frac_data = pd.read_csv('../../data/mass_fraction_compiled.csv')
# elong_data = pd.read_csv('../../data/dai2016_elongation_rate.csv')

# %%
# Set up the parameter values/ranges
gamma_max = 20 * 3600 / 7459
nu_max = np.linspace(0.01, 10, 300)
Kd = 0.02
phi_O = 0.35
phi_R_const = 0.1  


# Scenario 1 -- Constant allocation parameters
const_phiR_lam = growth.model.steady_state_growth_rate(gamma_max, nu_max, phi_R_const,
                                                       1-phi_O - phi_R_const, 
                                                       Kd)
const_phiR_tRNA = growth.model.steady_state_tRNA_balance(nu_max, 1 - phi_O - phi_R_const,
                                                         const_phiR_lam)
const_phiR_gamma = growth.model.translation_rate(gamma_max, const_phiR_tRNA, Kd)


# Scenario 2 -- Translation rate maximization
max_gamma_phi_R = growth.model.phi_R_max_translation(gamma_max, nu_max, phi_O)
max_gamma_lam = growth.model.steady_state_growth_rate(gamma_max, nu_max, max_gamma_phi_R,
                                                    1-phi_O-max_gamma_phi_R,
                                                      Kd)
max_gamma_tRNA = growth.model.steady_state_tRNA_balance(nu_max, 1 - phi_O - max_gamma_phi_R,
                                                        max_gamma_lam)
# Scenario 3 -- Growth rate maximization
opt_phi_R = growth.model.phi_R_optimal_allocation(gamma_max, nu_max, Kd, phi_O)
opt_allo_lam =  growth.model.steady_state_growth_rate(gamma_max, nu_max, opt_phi_R,
                                                      1 - phi_O - opt_phi_R,
                                                      Kd)
lam_range = np.linspace(0, 3,  300)
phiP = 1 - phi_O - opt_phi_R
opt_allo_lam_closed = - lam_range * (Kd * lam_range - lam_range + nu_max * phiP) / (gamma_max * (lam_range - nu_max * phiP))
opt_allo_tRNA = growth.model.steady_state_tRNA_balance(nu_max, 1 - phi_O - opt_phi_R,
                                                        opt_allo_lam)
opt_allo_gamma = growth.model.translation_rate(gamma_max, opt_allo_tRNA, Kd) 

fig, ax = plt.subplots(3, 3, figsize=(6, 4), sharex=True)

for i in range(3):
    ax[2, i].set_xlabel(r'$\nu_{max}\, /\, \gamma_{max}$')
    ax[0, i].set_ylabel('$\phi_R$')
    ax[1, i].set_ylabel('$\gamma\, /\, \gamma_{max}$')
    ax[2, i].set_ylabel('$\lambda\, /\, \gamma_{max}$')
    ax[2, i].set_ylim([0, 0.3]) 
    ax[0, i].set_ylim([0, 0.35]) 
    ax[1, i].set_ylim([0, 1.1]) 

# Growth rate vs nu
ax[2, 0].plot(nu_max/gamma_max, const_phiR_lam/gamma_max,  '-', lw=0.75, color=colors['primary_purple'])
ax[2, 1].plot(nu_max/gamma_max, max_gamma_lam/gamma_max, '-', lw=0.75, color=colors['primary_green'])
ax[2, 2].plot(nu_max/gamma_max, opt_allo_lam/gamma_max, '-', lw=0.75, color=colors['primary_blue'])

# PhiRr vs nu
ax[0, 0].plot(nu_max/gamma_max, phi_R_const * np.ones(len(nu_max)), '-', lw=0.75, color=colors['primary_purple'])
ax[0, 1].plot(nu_max/gamma_max, max_gamma_phi_R, '-', lw=0.75, color=colors['primary_green'])
ax[0, 2].plot(nu_max/gamma_max, opt_phi_R, '-', lw=0.75, color=colors['primary_blue'])

# Gamma vs nu
ax[1, 0].plot(nu_max/gamma_max, const_phiR_gamma/gamma_max, '-', lw=0.75, color=colors['primary_purple'])
ax[1, 1].plot(nu_max/gamma_max, np.ones(len(nu_max)), '-', lw=0.75, color=colors['primary_green'])
ax[1, 2].plot(nu_max/gamma_max, opt_allo_gamma/gamma_max, '-', lw=0.75, color=colors['primary_blue'])

plt.subplots_adjust(wspace=0.325, hspace=0.1)
plt.savefig('../../figures/regulatory_scenarios_theory.pdf', bbox_inches='tight')


# fig, ax = plt.subplots(1, 2, figsize=(4, 2))

# for a in ax:
#     a.set_xlabel('growth rate [hr$^{-1}$]')
# ax[0].set_ylabel('ribosomal mass fraction $\phi_R$')
# ax[1].set_ylabel('elongation rate [AA / sec]')

# # Mass fraction vs growth rate
# # ax[0].plot(lam_range, opt_allo_lam_closed, 'c-')

# ax[0].plot(const_phiR_lam, phi_R_const * np.ones(len(const_phiR_lam)), ':', lw=0.75, label='constant allocation')
# ax[0].plot(max_gamma_lam, max_gamma_phi_R, '--', lw=0.75, label='maximal elongation rate')
# ax[0].plot(opt_allo_lam, opt_phi_R, '-', lw=0.75, label='maximal growth rate')

# # Mass frac data 
# markers = ['X', 'o', 's', '^', 'v', '>']
# count = 0
# for g, d in frac_data.groupby('source'): 
#     ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], marker=markers[count], 
#                color=colors['primary_black'], linestyle='none', ms=3, alpha=0.75, 
#                label=g)
#     count += 1
# ax[0].legend(bbox_to_anchor = (3.1,1))

# ax[1].plot(const_phiR_lam, const_phiR_gamma, ':', lw=0.75, label='constant allocation')
# ax[1].plot(max_gamma_lam, gamma_max * np.ones(len(max_gamma_lam)) * (7459/3600), '--', lw=0.75, label='maximal elongation rate')
# ax[1].plot(opt_allo_lam, opt_allo_gamma, '-', lw=0.75, label='maximal growth rate')
# ax[1].plot(elong_data['growth_rate_hr'], elong_data['elongation_rate_aa_s'], 'o',
#             ms=3,
#             color=colors['primary_black'])


# plt.savefig('../../figures/reg_scenarios_with_data.pdf')

# %%
fig, ax = plt.subplots(1, 2)

