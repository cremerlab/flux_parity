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
phiRb_range = np.linspace(0.001, 0.999, 300)
palette = sns.color_palette('crest', len(nu_max))
grey_palette = sns.color_palette('Greys', len(nu_max) + 5)

# generate the figure
fig, ax = plt.subplots(2, 3, figsize=(6, 5))
for a in ax.ravel():
    a.set_xlabel('ribosomal allocation $\phi_{Rb}$', fontsize=8)

# Add axis labels
for i in range(2):
        ax[i,0].set(ylabel='$c_{pc}^* / K_D^{c_{pc}}$',
                  title='steady-state\nprecursor concentration $c_{pc}^*$',
                  yscale='log',
                  ylim=[0.1, 1E3])
        ax[i,1].set(ylabel='$\gamma$ [hr$^{-1}$]',
                 title='steady-state\ntranslation rate $\gamma$')
        ax[i,2].set(ylabel='$\lambda$ [hr$^{-1}$]')
        ax[i,2].set_title(label='steady-state growth rate $\lambda$', y=1.05)

# Add panel values
height=0.89
for i, nu in enumerate(nu_max):
    # Compute the steady state values
    params = (const['gamma_max'], phiRb_range, nu, const['Kd_cpc'])

    lam = growth.model.steady_state_growth_rate(*params)
    gamma = growth.model.steady_state_gamma(*params)
    cpc = growth.model.steady_state_precursors(*params)

    # Add glyphs
    for j, pal in enumerate([palette, grey_palette]):
        if j == 0:
            alpha = 1
        else: 
            alpha = 0.75        
        ax[j,0].plot(phiRb_range, cpc / const['Kd_cpc'], color=pal[i], lw=1, alpha=alpha)
        ax[j,1].plot(phiRb_range, gamma, color=pal[i], lw=1, alpha=alpha)
        ax[j,2].plot(phiRb_range, lam, color=pal[i], lw=1, alpha=alpha)


# Ccompute the various scenarios
nu_max = np.linspace(0, 5, 300)

# Scenario II: Constant gamma
phiRb_trans = nu_max / (const['gamma_max'] + nu_max)
strat2_params = (const['gamma_max'], phiRb_trans, nu_max, const['Kd_cpc'])
phiRb_trans_lam = growth.model.steady_state_growth_rate(*strat2_params)
phiRb_trans_cpc = growth.model.steady_state_precursors(*strat2_params)
phiRb_trans_gamma = growth.model.steady_state_gamma(*strat2_params)
ax[1, 0].plot(phiRb_trans, phiRb_trans_cpc / const['Kd_cpc'], lw=1, color=colors['primary_green'])
ax[1, 1].plot(phiRb_trans, phiRb_trans_gamma, lw=1, color=colors['primary_green'])
ax[1, 2].plot(phiRb_trans, phiRb_trans_lam, lw=1, color=colors['primary_green'])


# Scenario III: Optimal ribosomal allocation
phiRb_opt = growth.model.phi_R_optimal_allocation(const['gamma_max'], nu_max, const['Kd_cpc'])
strat3_params = (const['gamma_max'], phiRb_opt, nu_max, const['Kd_cpc']) 
phiRb_opt_lam = growth.model.steady_state_growth_rate(*strat3_params)
phiRb_opt_cpc = growth.model.steady_state_precursors(*strat3_params)
phiRb_opt_gamma = growth.model.steady_state_gamma(*strat3_params)
ax[1, 0].plot(phiRb_opt, phiRb_opt_cpc / const['Kd_cpc'], lw=1, color=colors['primary_blue'])
ax[1, 1].plot(phiRb_opt, phiRb_opt_gamma, lw=1, color=colors['primary_blue'])
ax[1, 2].plot(phiRb_opt, phiRb_opt_lam, lw=1, color=colors['primary_blue'])

# Scenario I: Constant allocation
phiRb_const = 0.2 * np.ones_like(nu_max)
strat1_params = (const['gamma_max'],  phiRb_const, nu_max, const['Kd_cpc'])
phiRb_const_lam = growth.model.steady_state_growth_rate(*strat1_params)
phiRb_const_cpc = growth.model.steady_state_precursors(*strat1_params)
phiRb_const_gamma = growth.model.steady_state_gamma(*strat1_params)
ax[1, 0].plot(phiRb_const, phiRb_const_cpc /const['Kd_cpc'], lw=1, color=colors['primary_purple'])
ax[1, 1].plot(phiRb_const, phiRb_const_gamma, lw=1, color=colors['primary_purple'])
ax[1, 2].plot(phiRb_const, phiRb_const_lam, lw=1, color=colors['primary_purple'])

# Tighten and save
plt.tight_layout()
plt.savefig('../../figures/Fig3_steady_state_plots.pdf', bbox_inches='tight')



# %%
