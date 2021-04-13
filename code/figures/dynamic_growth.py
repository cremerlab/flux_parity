#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate
import tqdm
import growth.viz 
import imp
import growth.model
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()
imp.reload(growth.model)

# Define the parameters
AVO = 6.022E23
VOL = 1E-3
OD_CONV = 1 / (7.7E-18)
gamma_max = 20 * 3600 / 7459 # Ribosomes per hr
Kd = 0.0013 * 20 
Km_0 = 1E5 # microgram glucose per liter

Km = (Km_0 * 1E-6) / (180.16)
phi_O = 0.35
phi_R = 0.3
phi_P = 1 - phi_R - phi_O
M0 = 0.1 * OD_CONV
M_P = phi_P * M0
M_R = phi_R * M0
m_AA = 0.01 * M0
m_N = 0.010 * 6.022E23 * 1E-3 
omega = 0.377 # * OD_CONV


dfs = []
nu_range = np.linspace(0.005, 10, 25)
time_range = np.linspace(0, 5, 300)
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    params = [M0, M_R, M_P, m_AA, m_N] 
    args = (gamma_max, nu, omega, phi_R, phi_P, Kd, Km)

    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator, 
                                params, time_range, args=args)
    _df = pd.DataFrame(out, columns=['biomass', 'ribos', 'metabs', 'm_aa', 'm_n'])
    _df['c_n'] = _df['m_n'].values / (6.022E23 * 1E-3)
    _df['m_n'] = _df['m_n'] / m_N
    _df['c_aa'] = _df['m_aa'].values / _df['biomass'].values
    _df['rel_biomass'] = _df['biomass'].values / M0
    _df['time'] = time_range * gamma_max
    _df['gamma'] =  (_df['c_aa'].values / (_df['c_aa'].values + Kd)) 
    _df['nu'] =  (_df['c_n'].values / (_df['c_n'].values + Km))
    _df['nu_max'] = nu
    dfs.append(_df)
df = pd.concat(dfs)
#%%
# Define the colors
cmap = sns.color_palette(f"mako_r", n_colors=len(nu_range) + 5)
cmap

fig, ax = plt.subplots(3, 2, figsize=(3.5, 5))
ax[0, 0].axis('off')
ax[0, 1].set_yscale('log')
for a in ax.ravel():
    a.set_xlabel(r'time $\times \gamma_{max}$')

# Add other lables
ax[0,1].set(title='biomass, $M$',
            ylabel=r'$M_t /  M_0$')
ax[1,0].set(title='translational capacity, $\gamma$',
            ylabel=r'$\gamma_t /  \gamma_{max}$')
ax[1,1].set(title=r'nutritional capacity, $\nu$',
            ylabel=r'$\nu_t /  \nu_{max}$')
ax[2,0].set(title='charged-tRNA concentration, $c_{AA}$',
            ylabel=r'$m_{AA} / M$')
ax[2,1].set(title='nutrient concentration, $c_N$',
            ylabel=r'$c_{N,t} / c_{N,0}$')

# Add panel lablels
fig.text(0.01, 0.99, '[A]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.99, '[B]', fontsize=8, fontweight='bold')
fig.text(0.01, 0.645, '[C]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.645, '[D]', fontsize=8, fontweight='bold')
fig.text(0.01, 0.32, '[E]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.32, '[F]', fontsize=8, fontweight='bold')

count = 0
for g, d in df.groupby('nu_max'):
    ax[0, 1].plot(d['time'], d['rel_biomass'], '-', color=cmap[count])
    ax[1, 0].plot(d['time'], d['gamma'], '-', color=cmap[count])
    ax[1, 1].plot(d['time'], d['nu'], '-', color=cmap[count])
    ax[2, 0].plot(d['time'], d['c_aa'], color=cmap[count])
    ax[2, 1].plot(d['time'], d['c_n'], color=cmap[count])
    count += 1
plt.tight_layout()
# plt.savefig('../../docs/figures/Fig2_integrated_dynamics_plots.pdf')

# %%
