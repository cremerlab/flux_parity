#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import matplotlib.pyplot as plt 
import growth.model
import growth.viz
import seaborn as sns
import tqdm
colors, palette = growth.viz.matplotlib_style()
np.random.seed(4310)
#%% Load the data
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[mass_frac['organism']=='Escherichia coli']
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong_rate = elong_rate[elong_rate['organism']=='Escherichia coli']
# tRNA = pd.read_csv('../../data/tRNA_abundances.csv')

# Map colors to sources
sources = []
for g, d in mass_frac.groupby(['source']):
    sources.append(g)
for g, d in elong_rate.groupby(['source']):
    if g not in sources:
        sources.append(g)
# palette = sns.color_palette('flare', n_colors=12).as_hex()
palette = sns.cubehelix_palette(start=.5, rot=-.5, n_colors=12).as_hex()
palette = np.random.choice(palette, len(sources), replace=False)
color_map = {s:c for s, c in zip(sources, palette)}
markers = ['o', 's', 'd', 'X', 'v', '^', '<', '>', 'P', 'p', 'h']
markers = np.random.choice(markers, len(sources), replace=False)
marker_map = {s:m for s, m in zip(sources, markers)}


#%%
# Define the parameters
gamma_max = 20 * 3600 / 7459
Kd_cpc = 0.01
nu_max = np.linspace(0.01, 10, 200)
Kd_TAA = 2E-5 #in M, Kd of uncharged tRNA to  ligase
Kd_TAA_star = 2E-5 # in M, Kd of charged tRNA to ribosom
kappa_max = (88 * 5 * 3600) / 1E9
tau = 3

# Compute the optimal scenario
opt_phiRb = growth.model.phi_R_optimal_allocation(gamma_max,  nu_max, Kd_cpc) 
opt_mu = growth.model.steady_state_growth_rate(gamma_max,  opt_phiRb, nu_max, Kd_cpc)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cpc) * 7459 / 3600

# Numerically compute the optimal scenario
dt = 0.0001
time_range = np.arange(0, 200, dt)
ss_df = pd.DataFrame([])
total_tRNA = 0.0004
T_AA = total_tRNA / 2
T_AA_star = total_tRNA / 2
for i, nu in enumerate(tqdm.tqdm(nu_max)):
    # Set the intitial state
    _opt_phiRb = growth.model.phi_R_optimal_allocation(gamma_max,  nu, Kd_cpc) 
    M0 = 1E9
    phi_Mb = 1 -  _opt_phiRb
    M_Rb = _opt_phiRb * M0
    M_Mb = phi_Mb * M0
    params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA_star, Kd_TAA, 0, kappa_max, 0, False, True, True)

    # Integrate
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    # Compute the final props
    _out = out[-1]
    ratio = _out[-1] / _out[-2]
    tRNA_abund = _out[-2] + _out[-1]
    biomass = _out[0]
    ss_df = ss_df.append({'phi_Rb': _out[1] / _out[0],
                          'lam': np.log(_out[0] / out[-2][0]) / (time_range[-1] - time_range[-2]),
                          'gamma': gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star) * 7459 / 3600,
                          'balance': ratio,
                          'tot_tRNA_abundance': tRNA_abund,
                          'biomass': biomass,
                          'tRNA_per_ribosome': (tRNA_abund * biomass) / (_out[1]/7459), 
                          'nu_max': nu},
                          ignore_index=True)

#%%
fig, ax = plt.subplots(1, 3, figsize=(6,2.5))
ax[0].axis('off')
ax[1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5], 
          xlabel='growth rate $\lambda$ [hr$^{-1}$]',
          ylabel='ribosomal allocation $\phi_{Rb}$')
ax[2].set(ylim=[5, 20], xlim=[-0.05, 2.5], xlabel='growth rate $\lambda$ [hr$^{-1}$]',
         ylabel='translation rate $\gamma$ [AA / s]')

for g, d in mass_frac.groupby('source'):
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'],  ms=4, marker=marker_map[g],
            color=color_map[g], label='__nolegend__', alpha=0.75,  markeredgecolor='k', markeredgewidth=0.25, linestyle='none')

for g, d in elong_rate.groupby(['source']):
    ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=marker_map[g], 
                 ms=4, alpha=0.75, color=color_map[g], markeredgecolor='k', markeredgewidth=0.25,
                 linestyle='none', label='__nolegend__')

# Plot the theory curves
ax[1].plot(opt_mu, opt_phiRb, '-', color=colors['primary_blue'], lw=1)
ax[1].plot(ss_df['lam'], ss_df['phi_Rb'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[2].plot(opt_mu, opt_gamma, '-', color=colors['primary_blue'], lw=1)
ax[2].plot(ss_df['lam'], ss_df['gamma'], '--', color=colors['primary_red'], zorder=1000, lw=1)
for s in sources:
    ax[0].plot([], [], ms=5, color=color_map[s], markeredgecolor='k',  markeredgewidth=0.25,
            marker=marker_map[s], label=s, linestyle='none')

ax[0].plot([], [], '-', color=colors['primary_blue'], lw=1, label='optimal allocation model')
ax[0].plot([], [], '--', color=colors['primary_red'], lw=1, label='ppGpp regulation model')
ax[0].legend()
plt.tight_layout()
plt.savefig('../../figures/Fig6_ppGpp_model_plots.pdf')


# %%
