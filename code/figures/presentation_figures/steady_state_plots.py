#%% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import seaborn as sns
colors, _ = growth.viz.matplotlib_style()


# Define the constants 
gamma_max = 20 * 3600 / 7459 # Ribosomes per hr
Kd = 0.02
phi_O = 0.35
phi_R = np.linspace(0.01, 1 - phi_O - 0.01, 200)
nu_range = np.linspace(0.01, 10, 25)
phi_P = 1 - phi_R - phi_O

# Compute the steady state solution
ss_df = []
for _, nu in enumerate(nu_range):
    lam_ss =  growth.model.steady_state_growth_rate(gamma_max, nu, phi_R, phi_P, Kd)
    cAA_ss = growth.model.sstRNA_balance(nu, phi_P, gamma_max, phi_R, Kd)
    gamma_ss = growth.model.translation_rate(gamma_max, cAA_ss, Kd)
    _df = pd.DataFrame([])
    _df['phiR'] = phi_R
    _df['lam'] = lam_ss / gamma_max
    _df['cAA'] = cAA_ss / Kd
    _df['gamma'] = gamma_ss / gamma_max
    _df['nu_max'] = nu
    ss_df.append(_df)
ss_df = pd.concat(ss_df, sort=False)
palette = sns.color_palette('mako_r', n_colors = len(nu_range))

# %%
fig, ax = plt.subplots(1, 3, figsize=(7, 2))
for a in ax:
    a.xaxis.set_tick_params(labelsize=6)
    a.yaxis.set_tick_params(labelsize=6)
    a.set_xlabel('ribosomal allocation $\phi_R$', fontsize=8)
ax[0].set_title('growth rate', fontsize=8)
ax[1].set_title('charged-tRNAs', fontsize=8)
ax[2].set_title('translational efficiency', fontsize=8)
ax[0].set_ylabel('$\lambda\, / \,  \gamma_{max}$', fontsize=8)
ax[1].set_ylabel('$c_{AA}\, / \, K_D$', fontsize=8)
ax[2].set_ylabel('$\gamma\, / \, \gamma_{max}$')
ax[1].set_yscale('log')

counter = 0
for g, d in ss_df.groupby(['nu_max']):

    ax[0].plot(d['phiR'], d['lam'], '-', lw=1, color=palette[counter])
    ax[1].plot(d['phiR'], d['cAA'], '-', lw=1, color=palette[counter])
    ax[2].plot(d['phiR'], d['gamma'], '-', lw=1, color=palette[counter])
    counter += 1
plt.savefig('../../../figures/presentations/steady_state_exploration.pdf', bbox_inches='tight')
# %%
