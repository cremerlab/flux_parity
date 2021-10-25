#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import seaborn as sns
import imp 
imp.reload(growth.model)
colors, _ = growth.viz.matplotlib_style()

# Set the constants
gamma_max = 20 * 3600/ 7459 
nu_max = np.linspace(0.1, 5, 20)
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3
phi_R = 0.5
phiP = 1 - phi_R
OD_CONV = 1.5E17

M0 = 0.001 * OD_CONV
Mr = phi_R * M0
Mp = phiP * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002
time_range = np.linspace(0, 150, 1500)
palette = sns.color_palette('crest', n_colors=len(nu_max) + 10)

dfs = []
for i, nu in enumerate(nu_max):
    params = [Mr, Mp, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA_star, Kd_TAA, False, True, kappa_max)
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    df= pd.DataFrame(out, columns=['Mr', 'Mp', 'T_AA', 'T_AA_star'])
    df['rel_biomass'] = (df['Mr'] + df['Mp']) / M0
    df['gamma'] = gamma_max * df['T_AA_star'].values / (df['T_AA_star'].values + Kd_TAA_star)
    df['nu'] = nu * df['T_AA'].values / (df['T_AA'].values + Kd_TAA)
    df['balance'] = df['T_AA_star'].values / df['T_AA'].values
    df['set_phiR'] = df['balance'].values / (df['balance'].values + tau)
    df['phi_R'] = df['Mr'].values / (df['Mr'].values + df['Mp'].values)
    df['nu_max'] = nu 
    df['time'] = time_range
    dfs.append(df)
data = pd.concat(dfs, sort=False)

fig, ax = plt.subplots(2, 3, figsize=(8, 6))
ax = ax.ravel()
for a in ax:
    a.set_xlabel('time [hr]')

ax[0].set(ylabel='relative biomass',
          yscale='log',
          title='biomass dynamics')
ax[1].set(ylabel=r'charged/uncharged',
          title='tRNA balance')
ax[2].set(ylabel=r'observed $\phi_R$',
          title='ribosomal allocation',
          ylim=[0, 1])
ax[3].set(ylabel=r'$\gamma / \gamma_{max}$',
          title='translational efficiency',
          ylim=[0, 1])
ax[4].set(ylabel=r'$\nu / \nu_{max}$',
          title='nutritional efficiency',
          ylim=[0, 1])
ax[5].set(ylabel='prescribed $\phi_R$',
          title='ribosome regulation',
          ylim=[0, 1])
counter = 0 
for g, d in data.groupby(['nu_max']):
    ax[0].plot(d['time'], d['rel_biomass'], '-', color=palette[counter])
    ax[1].plot(d['time'], d['balance'], '-', color=palette[counter])
    ax[2].plot(d['time'], d['phi_R'], '-', color=palette[counter])
    ax[3].plot(d['time'], d['gamma'] / gamma_max, '-', color=palette[counter])
    ax[4].plot(d['time'], d['nu'] / g, '-', color=palette[counter])
    ax[5].plot(d['time'], d['set_phiR'], '-', color=palette[counter])
    counter += 1

plt.tight_layout()
# plt.savefig('./ppGpp_integration_data_comparison.pdf')

#%%
# Compute the 'simple' model solutions
Kd = 0.012

nu_max = np.linspace(0.05, 12, 300)
optimal_phiRb = growth.model.phi_R_optimal_allocation(gamma_max, nu_max, Kd)
optimal_gamma = growth.model.steady_state_gamma(gamma_max, optimal_phiRb, nu_max, Kd) * 7459/3600
optimal_lam = growth.model.steady_state_growth_rate(gamma_max, optimal_phiRb, nu_max, Kd)
optimal_cpc = growth.model.steady_state_precursors(gamma_max, optimal_phiRb, nu_max, Kd)
m =1 
cpc = m * optimal_cpc
Nrb = m * optimal_phiRb / 7459
cpc_per_ribo = cpc / Nrb
# %%
# Set the constants
nu_max = np.linspace(0.05, 12, 300)
ss_df = pd.DataFrame([])
for i, nu in enumerate(nu_max):
    params = [Mr, Mp, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    _out = out[-1]
    ratio = _out[-1] / _out[-2]
    ss_df = ss_df.append({'phi_R': _out[0] / (_out[0]+ _out[1]),
                          'mu': np.log((_out[0] + _out[1]) / (out[-2][0] + out[-2][1])) / (time_range[-1] - time_range[-2]),
                          'gamma': gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star),
                          'balance': _out[-1] / (_out[-1] + _out[-2]),
                          'tRNA_per_ribosome': (_out[-1] + _out[-2]) * (_out[0] + _out[1]) / (_out[0]/7459),
                          'tot_tRNA': _out[-2] + _out[-1],
                          'nu_max': nu},
                          ignore_index=True)
# %%
# Load the elongation data
elong_data = pd.read_csv('../../data/peptide_elongation_rates.csv')
ecoli_elong = elong_data[elong_data['organism']=='Escherichia coli']
tRNA_data = pd.read_csv('../../data/tRNA_abundances.csv')
# Load the mass fraction data 
ecoli_mass = pd.read_csv('../../data/collated_mass_fraction_measurements.csv')

fig, ax = plt.subplots(2, 2, figsize=(6, 4))

for a in ax.ravel():
    a.set_xlim([0, 2.8])
for g, d in ecoli_mass.groupby(['source']):
    ax[0,0].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', ms=4)
for g, d in ecoli_elong.groupby(['source']):
    ax[0,1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o', ms=4)
for g, d in tRNA_data.groupby(['source']):
    ax[1, 0].plot(d['growth_rate_hr'], d['tRNA_per_ribosome'], 'o', ms=4)


# Plot ppGpp model
ax[0,0].plot(ss_df['mu'], ss_df['phi_R'], 'k-', lw=1)
ax[0,0].set_xlabel('growth rate µ [per hr]')
ax[0, 0].set_ylabel('$\phi_R$')
ax[0,1].plot(ss_df['mu'], ss_df['gamma'] * 7459 / 3600, 'k-', lw=1)
ax[0, 1].set_ylabel(r'$\gamma$ [AA/s]')
ax[0, 1].set_xlabel('growth rate µ [per hr]')
ax[1, 0].set_xlabel('growth rate µ [per hr]')
ax[1, 0].set_ylabel('tRNA per ribosome')
ax[1, 0].plot(ss_df['mu'], ss_df['tRNA_per_ribosome'], 'k-', lw=1)
ax[1,0].set_ylim([0, 30])
ax[1,1].plot(ss_df['mu'], ss_df['balance'], 'k-', lw=1) 
ax[1, 1].set_xlabel('growth rate µ [per hr]')
ax[1, 1].set_ylabel('charged fraction of tRNAs')

# Plot optimal results
ax[0,0].plot(optimal_lam, optimal_phiRb, '--',lw=1,  color=colors['primary_red'])
ax[0,1].plot(optimal_lam, optimal_gamma, '--', lw=1, color=colors['primary_red'])


plt.tight_layout()

# plt.savefig('../figures/ppGpp_data_comparison.pdf', bbox_inches='tight')

# %%

# %%
