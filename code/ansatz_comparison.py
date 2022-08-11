#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz
import growth.integrate
import tqdm
consts = growth.model.load_constants()
colors, palette = growth.viz.matplotlib_style()
source_colors = growth.viz.load_markercolors()

# Load the experimental data
mass_fractions = pd.read_csv('../data/main_figure_data/ecoli_ribosomal_mass_fractions.csv')
elongation_rate = pd.read_csv('../data/main_figure_data/ecoli_peptide_elongation_rates.csv')
ppGpp_data = pd.read_csv('../data/main_figure_data/ecoli_relative_ppGpp.csv')

# %%
# Define the constants
gamma_max = consts['gamma_max']
nu_range = np.linspace(0.1, 20, 300)
Kd_TAA = consts['Kd_TAA']
Kd_TAA_star = consts['Kd_TAA_star']
kappa_max = consts['kappa_max']
tau = consts['tau']
phi_O = consts['phi_O']
df = pd.DataFrame([])
for i in range(2):
    _df = pd.DataFrame([])
    for j, nu in enumerate(tqdm.tqdm(nu_range)):
        args = {'gamma_max': gamma_max,
                'nu_max':nu,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star': Kd_TAA_star,
                'tau': tau,
                'kappa_max':kappa_max,
                'phi_O': phi_O}
        if i == 0:
            args['ansatz'] = 'binding'
            ansatz = 'binding'
        else:
            ansatz = 'ratiometric'
        out = growth.integrate.equilibrate_FPM(args, max_iter=100)
        gamma = gamma_max * out[-1] / (out[-1] + Kd_TAA_star)
        balance = out[-1] / out[-2]
        if i == 0:
            phiRb =  (1 - phi_O) * out[-1] / (out[-2] + out[-1])
        else:
            phiRb = (1 - phi_O) * balance / (balance + tau)

        lam = gamma * phiRb
        __df = pd.DataFrame([np.array([phiRb, lam, gamma])], columns=['phiRb', 'growth_rate', 'gamma'])
        __df['TAA'] = out[-2]
        __df['TAA_star'] = out[-1]
        __df['ratio']= out[-1] / out[-2]
        __df['nu_max'] = nu
        __df['ansatz'] = ansatz
        _df = pd.concat([_df, __df])

    # Compute reference parameters for relative ppGpp
    ref_args = {'gamma_max':gamma_max, 
        'nu_max': 4,
        'Kd_TAA_star': Kd_TAA_star,
        'Kd_TAA': Kd_TAA,
        'phi_O': phi_O,
        'kappa_max': kappa_max,
        'tau': tau}
    if i == 0:
        ref_args['ansatz'] = 'binding'

    out = growth.integrate.equilibrate_FPM(ref_args)
    ref_ratio = out[-1]/out[-2]
    if i == 0:
        _df['rel_ppGpp'] = (1 + ref_ratio) / (1 + _df['ratio'])
    else:
        _df['rel_ppGpp'] = ref_ratio / _df['ratio']
    df = pd.concat([df, _df])


#%%
# Instantiate the figure and format the axes.
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[0].set_ylabel('$\phi_{Rb}$\n allocation towards ribosomes', fontsize=6)
ax[1].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6) 
ax[1].set_ylabel('$v_{tl}$ \ntranslation speed [AA/s]', fontsize=6)
ax[2].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[2].set_ylabel('[ppGpp] / [ppGpp]$_0$', fontsize=6)
ax[0].set_xlim([-0.1, 2.3])
ax[1].set_xlim([-0.1, 2.1])
ax[0].set_ylim([0, 0.3])
ax[1].set_ylim([5.5, 20])
ax[2].set_ylim([0.1, 35])
ax[2].set_yscale('log')
ax[2].set_xlim([0, 2.75])
ax[2].set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5])

# Plot the data
for g, d in mass_fractions.groupby(['source']):    
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=5,
            label='__nolegend__')

for g, d in elongation_rate.groupby(['source']):    
    ax[1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=5,
            label='__nolegend__')

# Plot the relative ppGpp curves
for g, d in ppGpp_data.groupby(['source']): 
    ax[2].plot(d['growth_rate_hr'], d['relative_ppGpp'], marker=source_colors[g]['m'],
            markerfacecolor=source_colors[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
            markersize=5, alpha=0.75, linestyle='none', label='__nolegend__')

# Plot the theory curves. 

for g, d in df.groupby(['ansatz']):
    if g == 'binding':
        c = colors['primary_blue']
        ls = '-'

    else:
        c= colors['primary_red']
        ls = '--'  
    g += ' ansatz'
 
    ax[0].plot(d['growth_rate'], d['phiRb'], label=g, lw=2, color=c, linestyle=ls)
    ax[1].plot(d['growth_rate'], d['gamma'] *  consts['m_Rb'] / 3600, label=g, lw=2, color=c, linestyle=ls)
    ax[2].plot(d['growth_rate'], d['rel_ppGpp'], label=g, color=c, linestyle=ls, lw=2) 

# Ad
ax[0].legend()
plt.tight_layout()
plt.savefig('../figures/FigSX_ansatz_comparison.pdf')

# %%

# %%
for g, d in df.groupby(['ansatz']):
    plt.plot((gamma_max / d['gamma']) - 1, d['rel_ppGpp'], label=g)
# plt.yscale('log')
plt.ylim([0, 8])
plt.xlabel('$\gamma_{max}/\gamma$ - 1')
plt.ylabel('relative ppGpp')
plt.legend()
# %%
