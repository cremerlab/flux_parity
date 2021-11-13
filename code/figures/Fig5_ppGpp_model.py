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
mapper = growth.viz.load_markercolors()
const = growth.model.load_constants()

#%% Load the data
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[mass_frac['organism']=='Escherichia coli']
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong_rate = elong_rate[elong_rate['organism']=='Escherichia coli']
tRNA = pd.read_csv('../../data/tRNA_abundances.csv')

#%%
# Define the parameters
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
nu_max = np.linspace(0.01, 10, 200)
Kd_TAA = 1E-5 #in M, Kd of uncharged tRNA to  ligase
Kd_TAA_star = 1E-5
kappa_max = const['kappa_max']
tau = const['tau']
phi_O = 0.25

# Compute the optimal scenario
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu_max, Kd_cpc, phi_O)
opt_mu = growth.model.steady_state_growth_rate(gamma_max,  opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cpc, phi_O) * 7459 / 3600

# Numerically compute the optimal scenario
dt = 0.0001
time_range = np.arange(0, 200, dt)
ss_df = pd.DataFrame([])
total_tRNA = 0.0004
T_AA = total_tRNA / 2
T_AA_star = total_tRNA / 2
for i, nu in enumerate(tqdm.tqdm(nu_max)):
    # Set the intitial state
    _opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu, Kd_cpc) 
    M0 = 1E9
    phi_Mb = 1 -  _opt_phiRb
    M_Rb = _opt_phiRb * M0
    M_Mb = phi_Mb * M0
    params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star, kappa_max, 0.25)

    # Integrate
    out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
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
fig, ax = plt.subplots(2, 2, figsize=(6,4.25))
ax[0, 0].axis('off')
ax[0, 1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5], 
          xlabel='growth rate $\lambda$ [hr$^{-1}$]',
          ylabel='ribosomal allocation $\phi_{Rb}$')
ax[1, 0].set(ylim=[5, 20], xlim=[-0.05, 2.5], xlabel='growth rate $\lambda$ [hr$^{-1}$]',
         ylabel='translation rate $\gamma$ [AA / s]')

ax[1, 1].set(ylim=[0, 22], xlabel='growth rate $\lambda$ [hr$^{-1}$]',
         ylabel='tRNA per ribosome')

sources = []
for g, d in mass_frac.groupby('source'):
    sources.append(g)
    ax[0, 1].plot(d['growth_rate_hr'], d['mass_fraction'],  ms=4, marker=mapper[g]['m'],
            color=mapper[g]['c'], label='__nolegend__', alpha=0.75,  markeredgecolor='k', 
            markeredgewidth=0.25, linestyle='none')

for g, d in elong_rate.groupby(['source']):
    if g not in sources:
        sources.append(g)
    ax[1, 0].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=mapper[g]['m'],
                 ms=4, alpha=0.75, color=mapper[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
                 linestyle='none', label='__nolegend__')

for g, d in tRNA.groupby(['source']):
    if g not in sources:
        sources.append(g)
    ax[1, 1].plot(d['growth_rate_hr'], d['tRNA_per_ribosome'], marker=mapper[g]['m'],
                ms=4, alpha=0.74, color=mapper[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
                linestyle='none')

# Plot the theory curves
ax[0, 1].plot(opt_mu, opt_phiRb, '-', color=colors['primary_blue'], lw=1)
ax[0, 1].plot(ss_df['lam'], ss_df['phi_Rb'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[1, 0].plot(opt_mu, opt_gamma, '-', color=colors['primary_blue'], lw=1)
ax[1, 0].plot(ss_df['lam'], ss_df['gamma'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[1, 1].plot(ss_df['lam'], ss_df['tRNA_per_ribosome'], '--', color=colors['primary_red'], zorder=1000,  lw=1)


for s in sources:
    if s == 'Bremer & Dennis, 1996':
        continue
    ax[0,0].plot([], [], ms=3, color=mapper[s]['c'], markeredgecolor='k',  markeredgewidth=0.25,
            marker=mapper[s]['m'], label=s, linestyle='none')

ax[0,0].legend()
plt.tight_layout()
plt.savefig('../../figures/Fig5_ppGpp_model_plots.pdf')


# %%
