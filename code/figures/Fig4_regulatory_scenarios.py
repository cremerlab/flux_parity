#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.viz
import seaborn as sns
colors, _ = growth.viz.matplotlib_style()

# Define the parameter ranges
gamma_max = 9.65
Kd_cAA = 0.025
nu_max_scenario = np.linspace(0, 5, 100)

nu_max = np.linspace(0, 5, 25)
palette = sns.color_palette('binary', n_colors=len(nu_max) + 5)
phiR_range = np.linspace(0.001, 0.999, 100)

fig, ax = plt.subplots(1, 3, figsize=(6, 2.5))

for a in ax.ravel():
    a.set_xlabel('ribosomal allocation $\phi_R$\n[mass abundance]', fontsize=8)

# Add axis labels
ax[0].set(ylabel='$\mu$ [hr$^{-1}$]')
        #   title='steady-state growth rate')
ax[0].set_title(label='steady-state growth rate $\mu$', y=1.05)
ax[1].set(ylabel='$c_{AA}$ [mass abundance]',
          title='steady-state\nprecursor abundance $c_{AA}$',
          yscale='log',
          ylim=[1E-3, 10])
ax[2].set(ylabel='$\gamma$ [hr$^{-1}$]',
         title='steady-state\ntranslational efficiency $\gamma$',
         )


# Compute the steady-state solutions

for i, nu in enumerate(nu_max):
    mu = growth.model.steady_state_mu(gamma_max, phiR_range, nu, Kd_cAA)
    gamma = growth.model.steady_state_gamma(gamma_max, phiR_range, nu, Kd_cAA)
    cAA = growth.model.steady_state_cAA(gamma_max, phiR_range, nu, Kd_cAA)

    # Add glyphs
    ax[0].plot(phiR_range, mu, color=palette[i], lw=0.75, alpha=0.75)
    ax[1].plot(phiR_range, cAA, color=palette[i], lw=0.75, alpha=0.75)
    ax[2].plot(phiR_range, gamma, color=palette[i], lw=0.75, alpha=0.75)

# Scenario I: Constant allocation
phiR_const = 0.25 * np.ones_like(nu_max_scenario)
phiR_const_mu = growth.model.steady_state_mu(gamma_max, phiR_const,nu_max_scenario, Kd_cAA)
phiR_const_cAA = growth.model.steady_state_cAA(gamma_max, phiR_const,nu_max_scenario, Kd_cAA)
phiR_const_gamma = growth.model.steady_state_gamma(gamma_max, phiR_const,nu_max_scenario, Kd_cAA)
ax[0].plot(phiR_const, phiR_const_mu, lw=1, color=colors['primary_purple'])
ax[1].plot(phiR_const, phiR_const_cAA, lw=1, color=colors['primary_purple'])
ax[2].plot(phiR_const, phiR_const_gamma, lw=1, color=colors['primary_purple'])

# Scenario II: Constant gamma
phiR_trans = nu_max_scenario / (gamma_max + nu_max_scenario)
phiR_trans_mu = growth.model.steady_state_mu(gamma_max, phiR_trans, nu_max_scenario, Kd_cAA)
phiR_trans_cAA = growth.model.steady_state_cAA(gamma_max, phiR_trans, nu_max_scenario, Kd_cAA)
phiR_trans_gamma = growth.model.steady_state_gamma(gamma_max, phiR_trans, nu_max_scenario, Kd_cAA)
ax[0].plot(phiR_trans, phiR_trans_mu, lw=1, color=colors['primary_green'])
ax[1].plot(phiR_trans, phiR_trans_cAA, lw=1, color=colors['primary_green'])
ax[2].plot(phiR_trans, phiR_trans_gamma, lw=1, color=colors['primary_green'])


# Scenario III: Optimal ribosomal allocation
phiR_opt = growth.model.phi_R_optimal_allocation(gamma_max, nu_max_scenario, Kd_cAA)
phiR_opt_mu = growth.model.steady_state_mu(gamma_max, phiR_opt, nu_max_scenario, Kd_cAA)
phiR_opt_cAA = growth.model.steady_state_cAA(gamma_max, phiR_opt, nu_max_scenario, Kd_cAA)
phiR_opt_gamma = growth.model.steady_state_gamma(gamma_max, phiR_opt, nu_max_scenario, Kd_cAA)
ax[0].plot(phiR_opt, phiR_opt_mu, lw=1, color=colors['primary_blue'])
ax[1].plot(phiR_opt, phiR_opt_cAA, lw=1, color=colors['primary_blue'])
ax[2].plot(phiR_opt, phiR_opt_gamma, lw=1, color=colors['primary_blue'])

# Tighten and save
plt.tight_layout()
plt.savefig('Fig4_steady_state_scenarios.pdf', bbox_inches='tight')


# %%
