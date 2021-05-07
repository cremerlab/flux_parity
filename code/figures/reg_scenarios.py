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
frac_data = pd.read_csv('../../data/mass_fraction_compiled.csv')
elong_data = pd.read_csv('../../data/dai2016_elongation_rate.csv')

# %%
# Set up the parameter values/ranges
gamma_max = 20 * 3600 / 7459
nu_max = np.linspace(0.01, 10, 300)
Kd = 0.02
phi_O = 0.35
phi_R_const = 0.10  


# Scenario 1 -- Constant allocation parameters
const_phiR_lam = growth.model.steady_state_growth_rate(gamma_max, nu_max, phi_R_const,
                                                       1-phi_O - phi_R_const, 
                                                       Kd)
const_phiR_tRNA = growth.model.steady_state_tRNA_balance(nu_max, 1 - phi_O - phi_R_const,
                                                         const_phiR_lam)
const_phiR_gamma = growth.model.translation_rate(gamma_max, const_phiR_tRNA, Kd) * 7459/3600


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
opt_allo_gamma = growth.model.translation_rate(gamma_max, opt_allo_tRNA, Kd) * 7459/3600

fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for a in ax:
    a.set_xlabel('growth rate [hr$^{-1}$]')
ax[0].set_ylabel('ribosomal mass fraction $\phi_R$')
ax[1].set_ylabel('elongation rate [AA / sec]')

# Mass fraction vs growth rate
# ax[0].plot(lam_range, opt_allo_lam_closed, 'c-')

ax[0].plot(const_phiR_lam, phi_R_const * np.ones(len(const_phiR_lam)), ':', lw=1, label='constant allocation')
ax[0].plot(max_gamma_lam, max_gamma_phi_R, '--', lw=1, label='maximal elongation rate')
ax[0].plot(opt_allo_lam, opt_phi_R, '-', lw=1, label='maximal growth rate')

# Mass frac data 
markers = ['X', 'o', 's', '^', 'v', '>']
count = 0
for g, d in frac_data.groupby('source'): 
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], marker=markers[count], 
               color=colors['primary_black'], linestyle='none', ms=3, alpha=0.75, 
               label=g)
    count += 1
ax[0].legend(bbox_to_anchor = (3.1,1))

ax[1].plot(const_phiR_lam, const_phiR_gamma, ':', lw=1, label='constant allocation')
ax[1].plot(max_gamma_lam, gamma_max * np.ones(len(max_gamma_lam)) * (7459/3600), '--', lw=1, label='maximal elongation rate')
ax[1].plot(opt_allo_lam, opt_allo_gamma, '-', lw=1, label='maximal growth rate')
ax[1].plot(elong_data['growth_rate_hr'], elong_data['elongation_rate_aa_s'], 'o',
            ms=3,
            color=colors['primary_black'])


plt.savefig('../../figures/reg_scenarios_with_data.pdf')

# %%
fig, ax = plt.subplots(1, 2)

