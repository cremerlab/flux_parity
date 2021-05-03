#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns 
import growth.viz 
import growth.model
colors, palette = growth.viz.matplotlib_style()

gamma_max = 20 * 3600 / 7459
nu_range = np.linspace(0.005, 10, 25)
phi_O = 0.35
phi_R = np.linspace(0.005, 1 - phi_O - 0.001, 300)
phi_P = 1 - phi_O - phi_R
Kd = 0.0265

ss_dfs = []
for i, nu in enumerate(nu_range):
    lam = growth.model.steady_state_growth_rate(gamma_max, nu, phi_R, phi_P, Kd)
    c_AA = growth.model.steady_state_tRNA_balance(nu, phi_P, lam)
    _df = pd.DataFrame([])
    _df['phi_R'] = phi_R
    _df['growth_rate'] = lam
    _df['c_AA'] = c_AA
    _df['gamma'] = growth.model.translation_rate(gamma_max, c_AA, Kd)
    _df['nu'] = nu
    ss_dfs.append(_df)
ss_df = pd.concat(ss_dfs, sort=False)

opt_df = pd.DataFrame([])
for i, nu in enumerate(nu_range):
    opt_phi_R = growth.model.optimal_phi_R(gamma_max, nu, Kd, phi_O, f_a=1)
    opt_phi_P = 1 - phi_O - opt_phi_R
    opt_lam = growth.model.steady_state_growth_rate(gamma_max, nu, opt_phi_R, opt_phi_P, Kd)
    opt_cAA = growth.model.steady_state_tRNA_balance(nu, opt_phi_P, opt_lam)
    gamma = growth.model.translation_rate(gamma_max, opt_cAA, Kd)
    opt_df = opt_df.append({'opt_phi_R':opt_phi_R, 
                            'nu_max':nu, 
                            'opt_growth_rate':opt_lam,
                            'opt_cAA':opt_cAA,
                            'opt_gamma':gamma},
                            ignore_index=True)

#%%
fig, ax = plt.subplots(1, 3, figsize=(5, 2))

# Add labels
ax[0].set_xlabel('ribosomal mass fraction $\phi_R$')
ax[0].set_ylabel('growth rate $\lambda$ [hr$^{-1}$]')
ax[1].set_xlabel('charged-tRNA abundance $c_{AA}$')
ax[1].set_ylabel('growth rate $\lambda$ [hr$^{-1}$]')
ax[2].set_xlabel('translational capacity $\gamma(c_{AA})$ [hr$^{-1}$]')
ax[2].set_ylabel('growth rate $\lambda$ [hr$^{-1}$]')

# Adjust scaling
ax[1].set_xscale('log')

line_palette = sns.color_palette('mako_r', n_colors=len(nu_range) + 5)
point_palette = sns.color_palette('magma', n_colors=len(nu_range) + 5)

counter = 0
for g, d in ss_df.groupby('nu'):
    ax[0].plot(d['phi_R'], d['growth_rate'], '-', lw=1, color=line_palette[counter])
    ax[1].plot(d['c_AA'], d['growth_rate'], '-', lw=1, color=line_palette[counter])
    ax[2].plot(d['gamma'], d['growth_rate'], '-', lw=1, color=line_palette[counter])
    counter += 1

counter =0
for g, d in opt_df.groupby('nu_max'):
    ax[0].plot(d['opt_phi_R'], d['opt_growth_rate'], 'o', markeredgecolor='w', 
                 ms=4, markeredgewidth=0.25, color=point_palette[counter])
    ax[1].plot(d['opt_cAA'], d['opt_growth_rate'], 'o',  ms=4, 
                markeredgecolor='w', markeredgewidth=0.25, color=point_palette[counter])
    ax[2].plot(d['opt_gamma'], d['opt_growth_rate'], 'o',  ms=4, 
               markeredgecolor='w', markeredgewidth=0.25, color=point_palette[counter])
    counter += 1

plt.tight_layout()
# plt.savefig('../../docs/figures/FigX_steady_state_plots.pdf')
# %%

data = pd.read_csv('../../../data/')