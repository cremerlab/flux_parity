#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import matplotlib.pyplot as plt 
import growth.model
import growth.viz
import tqdm
colors, palette = growth.viz.matplotlib_style()

#%% Load the data
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[mass_frac['organism']=='Escherichia coli']
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong_rate = elong_rate[elong_rate['organism']=='Escherichia coli']
tRNA = pd.read_csv('../../data/tRNA_abundances.csv')

# Define the parameters
gamma_max = 20 * 3600 / 7459
Kd_cAA = 0.01
nu_max = np.linspace(0.01, 10, 300)
Kd_TAA = 4E-5 #in M, Kd of uncharged tRNA to  ligase
Kd_TAA_star = 4E-5 # in M, Kd of charged tRNA to ribosom
tau = 3

# Compute the optimal scenario
opt_phiRb = growth.model.phi_R_optimal_allocation(gamma_max,  nu_max, Kd_cAA) 
opt_mu = growth.model.steady_state_mu(gamma_max,  opt_phiRb, nu_max, Kd_cAA)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cAA)

# Numerically compute the optimal scenario
time_range = np.linspace(0, 100, 5000)
ss_df = pd.DataFrame([])
total_tRNA = 0.0004
T_AA = total_tRNA / 2
T_AA_star = total_tRNA / 2
for i, nu in enumerate(tqdm.tqdm(nu_max)):
    # Set the intitial state
    _opt_phiRb = growth.model.phi_R_optimal_allocation(gamma_max,  nu, Kd_cAA) 
    M0 = 1E11
    phi_Mb = 1 -  _opt_phiRb
    M_Rb = _opt_phiRb * M0
    M_Mb = phi_Mb * M0
    params = [M_Rb, M_Mb, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star)

    # Integrate
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    # Compute the final props
    _out = out[-1]
    ratio = _out[-1] / _out[-2]
    tRNA_abund = _out[-2] + _out[-1]
    biomass = _out[0] + _out[1]
    ss_df = ss_df.append({'phi_Rb': _out[0] / (_out[0]+ _out[1]),
                          'mu': np.log((_out[0] + _out[1]) / (out[-2][0] + out[-2][1])) / (time_range[-1] - time_range[-2]),
                          'gamma': gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star),
                          'balance': ratio,
                          'tot_tRNA_abundance': tRNA_abund,
                          'biomass': biomass,
                          'tRNA_per_ribosome': (tRNA_abund * biomass) / (_out[0]/7459), 
                          'nu_max': nu},
                          ignore_index=True)


fig, ax = plt.subplots(1, 3, figsize=(6,2.5))
ax[0].axis('off')
ax[1].set(ylim=[0, 0.3], xlim=[-0.05, 2.5], 
          xlabel='growth rate µ [hr$^{-1}$]',
          ylabel='ribosomal allocation $\phi_{Rb}$')
ax[2].set(ylim=[3, 10], xlim=[-0.05, 2.5], xlabel='growth rate µ [hr$^{-1}$]',
         ylabel='translational efficiency $\gamma$ [hr$^{-1}$]')
# ax[2].set(xlim=[0,  3], ylim=[5, 20], xlabel='growth rate µ [hr$^{-1}$]',
#         ylabel='tRNA per ribosomes')

# Plot the data

# Define markers
markers = ['s', 'o', 'd', 'X', 'v', '^']

# Plot mass fraction
for g, d in mass_frac.groupby('organism'):
    counter = 0
    for _g, _d in d.groupby(['source']): 
        ax[1].plot(_d['growth_rate_hr'], _d['mass_fraction'], ms=5, marker=markers[counter],
                label='__nolegend__', alpha=0.75, linestyle='none')
        counter += 1

# For elongation rate, do specific mapping of colors and shapes
elong_map = {'Dai et al., 2016': [colors['primary_blue'], 'o'],
             'Forchammer & Lindahl, 1971': [colors['primary_green'], 'd'],
             'Bremmer & Dennis, 2008': [colors['primary_black'], 's'],
             'Dalbow and Young 1975' : [palette[-1], '>'],
             'Young and Bremer 1976' : [palette[-2], '<']}

for g, d in elong_rate.groupby(['source']):
    ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values * 3600 / 7459,
                 ms=5, marker=elong_map[g][1], color=elong_map[g][0],
                 linestyle='none', alpha=0.75, label='__nolegend__')

# for g, d in tRNA.groupby(['source']):
#     ax[2].plot(d['growth_rate_hr'], d['tRNA_per_ribosome'], 'o')

# ax[2].plot(ss_df['mu'], ss_df['tRNA_per_ribosome'], '--', color=colors['primary_red'], lw=1)


# Plot the theory curves
ax[1].plot(opt_mu, opt_phiRb, '-', color=colors['primary_blue'], lw=1)
ax[1].plot(ss_df['mu'], ss_df['phi_Rb'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[2].plot(opt_mu, opt_gamma, '-', color=colors['primary_blue'], lw=1)
ax[2].plot(ss_df['mu'], ss_df['gamma'], '--', color=colors['primary_red'], zorder=1000, lw=1)
plt.tight_layout()
plt.savefig('../../figures/Fig6_ppGpp_model_plots.pdf')
# %%

# %%
