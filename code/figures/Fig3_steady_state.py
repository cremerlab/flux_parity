#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import seaborn as sns
colors, _  = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Define the parameter ranges
nu_max = np.linspace(0, 5, 10)
phi_O = 0.25
phiRb_range = np.linspace(0.001, 1 - phi_O - 0.001, 300)

palette = sns.color_palette('crest', len(nu_max))

# generate the figure
fig, ax = plt.subplots(2, 3, figsize=(6.5, 4))
for i in range(3):
    ax[0, i].set_xlabel('ribosomal allocation $\phi_{Rb}$', fontsize=8)
    ax[1, i].set_xlabel(r'maximum metabolic rate $\nu_{max}$ [hr$^{-1}$]', fontsize=8)
    ax[0, i].set_xlim([0, 0.75])

# Add axis labels
ax[0, 0].set(ylabel='$c_{pc}^* / K_D^{c_{pc}}$', yscale='log')
ax[1, 0].set_ylabel('$\phi_{Rb}$')
for i in range(2):
        ax[i,1].set(ylabel='$\gamma$ [hr$^{-1}$]')
        ax[i,2].set(ylabel='$\lambda$ [hr$^{-1}$]')


# Add panel values
height=0.89
for i, nu in enumerate(nu_max):
    # Compute the steady state values
    params = (const['gamma_max'], phiRb_range, nu, const['Kd_cpc'], phi_O)

    lam = growth.model.steady_state_growth_rate(*params)
    gamma = growth.model.steady_state_gamma(*params)
    cpc = growth.model.steady_state_precursors(*params)

    # Add glyphs
    if nu == nu_max[int(len(nu_max) / 2)]:
        color = colors['primary_red']
        zorder = 1000
        lw = 1.5
    else:
        color = palette[i]
        zorder = 1
        lw = 1
    ax[0,0].plot(phiRb_range, cpc / const['Kd_cpc'], color=color, lw=lw, zorder=zorder)
    ax[0,1].plot(phiRb_range, gamma, color=color, lw=lw, zorder=zorder)
    ax[0,2].plot(phiRb_range, lam, color=color, lw=lw, zorder=zorder)

# Ccompute the various scenarios
nu_max = np.linspace(0, 5, 300)

# Scenario II: Constant gamma
phiRb_trans = (nu_max * (1 - phi_O)) / (const['gamma_max'] + nu_max)
strat2_params = (const['gamma_max'], phiRb_trans, nu_max, const['Kd_cpc'], phi_O)
phiRb_trans_lam = growth.model.steady_state_growth_rate(*strat2_params)
phiRb_trans_cpc = growth.model.steady_state_precursors(*strat2_params)
phiRb_trans_gamma = growth.model.steady_state_gamma(*strat2_params)
ax[1, 0].plot(nu_max, phiRb_trans, lw=1, color=colors['primary_green'])
ax[1, 1].plot(nu_max, phiRb_trans_gamma, lw=1, color=colors['primary_green'])
ax[1, 2].plot(nu_max, phiRb_trans_lam, lw=1, color=colors['primary_green'])


# Scenario III: Optimal ribosomal allocation
phiRb_opt = growth.model.phi_R_optimal_allocation(const['gamma_max'], nu_max, const['Kd_cpc'], phi_O)
strat3_params = (const['gamma_max'], phiRb_opt, nu_max, const['Kd_cpc'], phi_O) 
phiRb_opt_lam = growth.model.steady_state_growth_rate(*strat3_params)
phiRb_opt_cpc = growth.model.steady_state_precursors(*strat3_params)
phiRb_opt_gamma = growth.model.steady_state_gamma(*strat3_params)
ax[1, 0].plot(nu_max, phiRb_opt, lw=1, color=colors['primary_blue'])
ax[1, 1].plot(nu_max, phiRb_opt_gamma, lw=1, color=colors['primary_blue'])
ax[1, 2].plot(nu_max, phiRb_opt_lam, lw=1, color=colors['primary_blue'])

# Scenario I: Constant allocation
phiRb_const = 0.1 * np.ones_like(nu_max)
strat1_params = (const['gamma_max'],  phiRb_const, nu_max, const['Kd_cpc'], phi_O)
phiRb_const_lam = growth.model.steady_state_growth_rate(*strat1_params)
phiRb_const_cpc = growth.model.steady_state_precursors(*strat1_params)
phiRb_const_gamma = growth.model.steady_state_gamma(*strat1_params)
ax[1, 0].plot(nu_max, phiRb_const, lw=1, color=colors['primary_black'])
ax[1, 1].plot(nu_max, phiRb_const_gamma, lw=1, color=colors['primary_black'])
ax[1, 2].plot(nu_max, phiRb_const_lam, lw=1, color=colors['primary_black'])

# Tighten and save
plt.tight_layout()
plt.savefig('../../figures/Fig3_steady_state_plots.pdf', bbox_inches='tight')



# %%
nu_max = np.linspace(0, 5, 10)
phi_O = 0.25
phiRb_range = np.linspace(0.001, 1 - phi_O - 0.001, 300)

# generate the figure
palette = sns.color_palette('Greys', n_colors=len(nu_max))
fig, ax = plt.subplots(1, 3, figsize=(6.5, 2))
for i in range(3):
    ax[i].set_xlabel('ribosomal allocation $\phi_{Rb}$', fontsize=8)

# Add axis labels
ax[0].set(ylabel='$c_{pc}^* / K_D^{c_{pc}}$', yscale='log')
ax[1].set(ylabel='$\gamma$ [hr$^{-1}$]')
ax[2].set(ylabel='$\lambda$ [hr$^{-1}$]')

# Add panel values
height=0.89
for i, nu in enumerate(nu_max):
    # Compute the steady state values
    params = (const['gamma_max'], phiRb_range, nu, const['Kd_cpc'], phi_O)

    lam = growth.model.steady_state_growth_rate(*params)
    gamma = growth.model.steady_state_gamma(*params)
    cpc = growth.model.steady_state_precursors(*params)

    # Add glyphs
    color = palette[i]
    zorder = 1
    lw = 1
    ax[0].plot(phiRb_range, cpc / const['Kd_cpc'], color=color, lw=lw, zorder=zorder, alpha=0.75)
    ax[1].plot(phiRb_range, gamma, color=color, lw=lw, zorder=zorder, alpha=0.75)
    ax[2].plot(phiRb_range, lam, color=color, lw=lw, zorder=zorder, alpha=0.75)

# Ccompute the various scenarios
nu_max = np.linspace(0, 5, 300)

# Scenario II: Constant gamma
phiRb_trans = (nu_max * (1 - phi_O)) / (const['gamma_max'] + nu_max)
strat2_params = (const['gamma_max'], phiRb_trans, nu_max, const['Kd_cpc'], phi_O)
phiRb_trans_lam = growth.model.steady_state_growth_rate(*strat2_params)
phiRb_trans_cpc = growth.model.steady_state_precursors(*strat2_params)
phiRb_trans_gamma = growth.model.steady_state_gamma(*strat2_params)
ax[0].plot(phiRb_trans, phiRb_trans_cpc / const['Kd_cpc'], lw=1, color=colors['primary_green'])
ax[1].plot(phiRb_trans, phiRb_trans_gamma, lw=1, color=colors['primary_green'])
ax[2].plot(phiRb_trans, phiRb_trans_lam, lw=1, color=colors['primary_green'])


# Scenario III: Optimal ribosomal allocation
phiRb_opt = growth.model.phi_R_optimal_allocation(const['gamma_max'], nu_max, const['Kd_cpc'], phi_O)
strat3_params = (const['gamma_max'], phiRb_opt, nu_max, const['Kd_cpc'], phi_O) 
phiRb_opt_lam = growth.model.steady_state_growth_rate(*strat3_params)
phiRb_opt_cpc = growth.model.steady_state_precursors(*strat3_params)
phiRb_opt_gamma = growth.model.steady_state_gamma(*strat3_params)
ax[0].plot(phiRb_opt, phiRb_opt_cpc / const['Kd_cpc'], lw=1, color=colors['primary_blue'])
ax[1].plot(phiRb_opt, phiRb_opt_gamma, lw=1, color=colors['primary_blue'])
ax[2].plot(phiRb_opt, phiRb_opt_lam, lw=1, color=colors['primary_blue'])

# Scenario I: Constant allocation
phiRb_const = 0.1 * np.ones_like(nu_max)
strat1_params = (const['gamma_max'],  phiRb_const, nu_max, const['Kd_cpc'], phi_O)
phiRb_const_lam = growth.model.steady_state_growth_rate(*strat1_params)
phiRb_const_cpc = growth.model.steady_state_precursors(*strat1_params)
phiRb_const_gamma = growth.model.steady_state_gamma(*strat1_params)
ax[0].plot(phiRb_const, phiRb_const_cpc / const['Kd_cpc'], lw=1, color=colors['primary_black'])
ax[1].plot(phiRb_const, phiRb_const_gamma, lw=1, color=colors['primary_black'])
ax[2].plot(phiRb_const, phiRb_const_lam, lw=1, color=colors['primary_black'])

# Tighten and save
plt.tight_layout()
plt.savefig('../../figures/Fig3_steady_state_plots_strategies.pdf', bbox_inches='tight')


# %%
