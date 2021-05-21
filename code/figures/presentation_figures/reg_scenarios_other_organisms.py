#%%
import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
colors, palette = growth.viz.matplotlib_style()

#%%
SHOW_DATA = False

# Load the relevant data 
marip = pd.read_csv('../../../data/Mueller2021.csv')
# elong_data = pd.read_csv('../../../data/dai2016_elongation_rate.csv')
#%%

# Set up the parameter values/ranges
gamma_max = 5 * 3600 / 7459
nu_max = np.linspace(0.01, 10, 300)
Kd = 0.02
phi_O = 0.7  
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

fig, ax = plt.subplots(1, 2, figsize=(6, 2))

for a in ax:
    a.set_xlabel('growth rate [hr$^{-1}$]')
ax[0].set_ylabel('ribosomal mass fraction $\phi_R$')
ax[1].set_ylabel('elongation rate [AA / sec]')

ax[0].plot(const_phiR_lam, phi_R_const * np.ones(len(const_phiR_lam)), lw=1, 
            color=colors['primary_purple'], label='constant allocation')
ax[0].plot(max_gamma_lam, max_gamma_phi_R, lw=1, label='maximal elongation rate',
            color=colors['primary_green'])
ax[0].plot(opt_allo_lam, opt_phi_R, lw=1, label='maximal growth rate',
            color=colors['primary_blue'])

# Mass frac data 
markers = ['X', 'o', 's', '^', 'v', '>']
count = 0
ax[0].plot(marip['growth_rate_hr'], marip['mass_frac'], 'o', label='Müller & Gu et al. 2021')

# for g, d in frac_data.groupby('source'): 
#     ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], marker=markers[count], 
#                    color=colors['primary_black'], linestyle='none', ms=5, alpha=0.75, 
#                    label=g, zorder=1000)
#         count += 1
#     ax[1].plot(elong_data['growth_rate_hr'], elong_data['elongation_rate_aa_s'], 'o',
#                 ms=5, color=colors['primary_black'], alpha=0.75, zorder=1000)

   
ax[1].plot(const_phiR_lam, const_phiR_gamma * (7459/3600), '-', lw=1, color=colors['primary_purple'])
ax[1].plot(max_gamma_lam, gamma_max * np.ones(len(max_gamma_lam)) * (7459/3600), lw=1, color=colors['primary_green'], label='maximal elongation rate')
ax[1].plot(opt_allo_lam, opt_allo_gamma * (7459/3600), '-', lw=1, color=colors['primary_blue'], label='maximal growth rate')
ax[0].legend(fontsize=5)
plt.subplots_adjust(wspace=0.3)
plt.savefig('../../../figures/presentations/maripaludis_data.pdf')
# if SHOW_DATA:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_data.pdf')
# else:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_nodata.pdf')
# %%
#%%


# Load the relevant data 
crassa = pd.read_csv('../../../data/Alberghina1974.csv')
# elong_data = pd.read_csv('../../../data/dai2016_elongation_rate.csv')


# Set up the parameter values/ranges
gamma_max = 15 * 3600 / 1E4 
nu_max = np.linspace(0.01, 10, 300)
Kd = 0.02
phi_O = 0.3  
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

fig, ax = plt.subplots(1, 2, figsize=(6, 2))

for a in ax:
    a.set_xlabel('growth rate [hr$^{-1}$]')
ax[0].set_ylabel('ribosomal mass fraction $\phi_R$')
ax[1].set_ylabel('elongation rate [AA / sec]')

ax[0].plot(const_phiR_lam, phi_R_const * np.ones(len(const_phiR_lam)), lw=1, 
            color=colors['primary_purple'], label='constant allocation')
ax[0].plot(max_gamma_lam, max_gamma_phi_R, lw=1, label='maximal elongation rate',
            color=colors['primary_green'])
ax[0].plot(opt_allo_lam, opt_phi_R, lw=1, label='maximal growth rate',
            color=colors['primary_blue'])
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 0.3])
ax[1].set_xlim([0, 1])
# Mass frac data 
markers = ['X', 'o', 's', '^', 'v', '>']
count = 0
ax[0].plot(crassa['growth_rate'], crassa['mass_fraction'], 'o', label='Alberghina 1974')
ax[1].plot(crassa['growth_rate'], crassa['elongation_rate'], 'o', label='Alberghina 1974')

# for g, d in frac_data.groupby('source'): 
#     ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], marker=markers[count], 
#                    color=colors['primary_black'], linestyle='none', ms=5, alpha=0.75, 
#                    label=g, zorder=1000)
#         count += 1
#     ax[1].plot(elong_data['growth_rate_hr'], elong_data['elongation_rate_aa_s'], 'o',
#                 ms=5, color=colors['primary_black'], alpha=0.75, zorder=1000)

   
ax[1].plot(const_phiR_lam, const_phiR_gamma * (1E4/3600), '-', lw=1, color=colors['primary_purple'])
ax[1].plot(max_gamma_lam, gamma_max * np.ones(len(max_gamma_lam)) * (1E4/3600), lw=1, color=colors['primary_green'], label='maximal elongation rate')
ax[1].plot(opt_allo_lam, opt_allo_gamma * (1E4/3600), '-', lw=1, color=colors['primary_blue'], label='maximal growth rate')
ax[0].legend(fontsize=5)
plt.subplots_adjust(wspace=0.3)
plt.savefig('../../../figures/presentations/crassa_data.pdf')
# if SHOW_DATA:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_data.pdf')
# else:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_nodata.pdf')
# %%

coel = pd.read_csv('../../../data/Cox2004_Table3-4.csv')
coel = coel[coel['organism']=='Streptomyces coelicolor']

#%%

# Set up the parameter values/ranges
gamma_max = 5 * 3600 / 7459 
nu_max = np.linspace(0.01, 10, 300)
Kd = 0.05
phi_O = 0.3
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

fig, ax = plt.subplots(1, 2, figsize=(6, 2))

for a in ax:
    a.set_xlabel('growth rate [hr$^{-1}$]')
ax[0].set_ylabel('ribosomal mass fraction $\phi_R$')
ax[1].set_ylabel('elongation rate [AA / sec]')

ax[0].plot(const_phiR_lam, phi_R_const * np.ones(len(const_phiR_lam)), lw=1, 
            color=colors['primary_purple'], label='constant allocation')
ax[0].plot(max_gamma_lam, max_gamma_phi_R, lw=1, label='maximal elongation rate',
            color=colors['primary_green'])
ax[0].plot(opt_allo_lam, opt_phi_R, lw=1, label='maximal growth rate',
            color=colors['primary_blue'])
ax[0].set_xlim([0, 1])
ax[0].set_ylim([0, 0.3])
ax[1].set_xlim([0, 1])
# Mass frac data 
markers = ['X', 'o', 's', '^', 'v', '>']
count = 0
ax[0].plot(coel['growth_rate_hr'], coel['ribosomal_mass_fraction'], 'o', label='Cox 2004')
ax[1].plot(coel['growth_rate_hr'], coel['elongation_rate_aa_sec'], 'o', label='Cox 2004')

   
ax[1].plot(const_phiR_lam, const_phiR_gamma * (7459/3600), '-', lw=1, color=colors['primary_purple'])
ax[1].plot(max_gamma_lam, gamma_max * np.ones(len(max_gamma_lam)) * (7459/3600), lw=1, color=colors['primary_green'], label='maximal elongation rate')
ax[1].plot(opt_allo_lam, opt_allo_gamma * (7459/3600), '-', lw=1, color=colors['primary_blue'], label='maximal growth rate')
ax[0].legend(fontsize=5)
plt.subplots_adjust(wspace=0.3)
plt.savefig('../../../figures/presentations/coelicolor_data.pdf')
# if SHOW_DATA:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_data.pdf')
# else:
#     plt.savefig('../../../figures/presentations/ecoli_regulatory_scenarios_nodata.pdf')
# %%
