#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import seaborn as sns
colors, _  = growth.viz.matplotlib_style()

# Define the parameter ranges
nu_max = np.linspace(0, 5, 25)
phiR_range = np.linspace(0.001, 0.999, 300)
palette = sns.color_palette('crest', len(nu_max))

# Define the constants 
Kd_cAA = 0.025
gamma_max = 9.65

# generate the figure
fig, ax = plt.subplots(1, 3, figsize=(6, 2.5))
for a in ax.ravel():
    a.set_xlabel('ribosomal allocation $\phi_R$\n[mass abundance]', fontsize=8)

# Add axis labels
ax[0].set(ylabel='$\mu$ [hr$^{-1}$]')
        #   title='steady-state growth rate')
ax[0].set_title(label='steady-state growth rate $\mu$', y=1.05)
ax[1].set(ylabel='$c_{AA}$ [mass abundance]',
          title='steady-state\nprecursor abundance $c_{AA}$',
          yscale='log')

ax[2].set(ylabel='$\gamma$ [hr$^{-1}$]',
         title='steady-state\ntranslational efficiency $\gamma$')

# Add panel values
height=0.89
fig.text(0.02, height, '(A)', fontsize=8, color=colors['primary_black'],
        fontweight='bold')

fig.text(0.335, height, '(B)', fontsize=8, color=colors['primary_black'],
        fontweight='bold')

fig.text(0.675, height, '(C)', fontsize=8, color=colors['primary_black'],
        fontweight='bold')

for i, nu in enumerate(nu_max):
    # Compute the steady state values
    mu = growth.model.steady_state_mu(gamma_max, phiR_range, nu, Kd_cAA)
    gamma = growth.model.steady_state_gamma(gamma_max, phiR_range, nu, Kd_cAA)
    cAA = growth.model.steady_state_cAA(gamma_max, phiR_range, nu, Kd_cAA)

    # Add glyphs
    ax[0].plot(phiR_range, mu, color=palette[i], lw=0.75)
    ax[1].plot(phiR_range, cAA, color=palette[i], lw=0.75)
    ax[2].plot(phiR_range, gamma, color=palette[i], lw=0.75)

# Tighten and save
plt.tight_layout()
plt.savefig('../../figures/Fig3_steady_state_plots.pdf', bbox_inches='tight')



# %%
