#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import seaborn as sns
colors, _ = growth.viz.matplotlib_style()

# Set the constants
gamma_max = 9.65
nu_max = np.linspace(0.1, 10, 20)
Kd_TAA = 2E-5#in M, Kd of uncharged tRNA to ligase
Kd_TAA_star = 2E-5 # in M, Kd of charged tRNA to ribosom
tau = 3 
phi_R = 0.5
phiP = 1 - phi_R
OD_CONV = 1.5E17

M0 = 0.001 * OD_CONV
Mr = phi_R * M0
Mp = phiP * M0
T_AA = 0.0002
T_AA_star = 0.0002
time_range = np.linspace(0, 100, 1000)
palette = sns.color_palette('crest', n_colors=len(nu_max))

dfs = []
for i, nu in enumerate(nu_max):
    params = [Mr, Mp, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star)
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    df= pd.DataFrame(out, columns=['Mr', 'Mp', 'T_AA', 'T_AA_star'])
    df['rel_biomass'] = (df['Mr'] + df['Mp']) / M0
    df['gamma'] = gamma_max * df['T_AA_star'].values / (df['T_AA_star'].values + Kd_TAA_star)
    df['nu'] = nu * df['T_AA'].values / (df['T_AA'].values + Kd_TAA)
    df['balance'] = df['T_AA_star'].values / df['T_AA'].values
    df['fa'] = df['balance'].values / (df['balance'].values + tau)
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
ax[2].set(ylabel=r'$\phi_R$',
          title='ribosomal allocation',
          ylim=[0, 1])
ax[3].set(ylabel=r'$\gamma / \gamma_{max}$',
          title='translational efficiency',
          ylim=[0, 1])
ax[4].set(ylabel=r'$\nu / \nu_{max}$',
          title='nutritional efficiency',
          ylim=[0, 1])
ax[5].set(ylabel='active fraction',
          title='ribosome regulation',
          ylim=[0, 1])
counter = 0 
for g, d in data.groupby(['nu_max']):
    ax[0].plot(d['time'], d['rel_biomass'], '-', color=palette[counter])
    ax[1].plot(d['time'], d['balance'], '-', color=palette[counter])
    ax[2].plot(d['time'], d['phi_R'], '-', color=palette[counter])
    ax[3].plot(d['time'], d['gamma'] / gamma_max, '-', color=palette[counter])
    ax[4].plot(d['time'], d['nu'] / g, '-', color=palette[counter])
    ax[5].plot(d['time'], d['fa'], '-', color=palette[counter])
    counter += 1

plt.tight_layout()
# plt.savefig('./ppGpp_integration_firstpass.pdf')
# %%


# %%

# %%
# Set the constants

nu_max = np.linspace(0.1, 10, 500)
ss_df = pd.DataFrame([])
for i, nu in enumerate(nu_max):
    params = [Mr, Mp, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star)
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    _out = out[-1]
    ratio = _out[-1] / _out[-2]
    ss_df = ss_df.append({'phi_R': _out[0] / (_out[0]+ _out[1]),
                          'mu': np.log((_out[0] + _out[1]) / (out[-2][0] + out[-2][1])) / (time_range[-1] - time_range[-2]),
                          'gamma': gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star),
                          'fa': ratio / (ratio + tau),
                          'balance': ratio,
                          'nu_max': nu},
                          ignore_index=True)
# %%

fig, ax = plt.subplots(2, 2, figsize=(6, 4))
for a in ax.ravel():
    a.set_xlabel('steady state $\phi_R$')

ax[0,0].plot(ss_df['mu'], ss_df['phi_R'], '-')
ax[0,0].set_ylabel('growth rate µ [per hr]')
ax[0,1].plot(ss_df['phi_R'], ss_df['gamma'], '-')
ax[0, 1].set_ylabel(r'$\gamma$ [per hr]')
ax[1,0].plot(ss_df['phi_R'], ss_df['fa'], '-')
ax[1, 0].set_ylabel('active fraction')
ax[1,1].plot(ss_df['phi_R'], ss_df['balance'], '-')
ax[1, 1].set_ylabel('charged/uncharged')

# %%
