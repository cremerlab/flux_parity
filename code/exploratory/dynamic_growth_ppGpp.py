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
nu_max = np.linspace(0.5, 5, 20)
Kd_TAA = 0.0002 #in M, Kd of uncharged tRNA to ligase
Kd_TAA_star = 0.00002 # in M, Kd of charged tRNA to ribosom
tau = 0.5
phi_R = 0.5
phiP = 1 - phi_R
OD_CONV = 1.5E17

M0 = 0.001 * OD_CONV
Mr = phi_R * M0
Mp = phiP * M0
T_AA = 0.001
T_AA_star = 0.0001 
time_range = np.linspace(0, 4, 1000)
palette = sns.color_palette('crest', n_colors=len(nu_max))

dfs = []
for i, nu in enumerate(nu_max):
    params = [Mr, Mp, T_AA, T_AA_star]
    args = (gamma_max, nu, phi_R, tau, Kd_TAA, Kd_TAA_star)
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                params, time_range,  args=args)
    df= pd.DataFrame(out, columns=['Mr', 'Mp', 'T_AA', 'T_AA_star'])
    df['rel_biomass'] = (df['Mr'] + df['Mp']) / M0
    df['gamma'] = gamma_max * df['T_AA_star'].values / (df['T_AA_star'].values + Kd_TAA_star)
    df['nu'] = nu * df['T_AA'].values / (df['T_AA'].values + Kd_TAA)
    df['balance'] = df['T_AA_star'].values / df['T_AA'].values
    df['fa'] = df['balance'].values / (df['balance'].values + tau)
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
        #   yscale='log',
          title='tRNA balance')
ax[2].axis('off')
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
    ax[3].plot(d['time'], d['gamma'] / gamma_max, '-', color=palette[counter])
    ax[4].plot(d['time'], d['nu'] / g, '-', color=palette[counter])
    ax[5].plot(d['time'], d['fa'], '-', color=palette[counter])
    counter += 1

plt.tight_layout()
plt.savefig('./ppGpp_integration_firstpass.pdf')
# %%

# %%

# %%
