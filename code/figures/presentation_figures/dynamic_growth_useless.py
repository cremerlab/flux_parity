#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate
import tqdm
import growth.viz 
import growth.model
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()

# Define the parameters
AVO = 6.022E23
VOL = 1E-3
OD_CONV = 1 / (7.7E-18)
gamma_max = 20 * 3600 / 7459 # Ribosomes per hr
Kd = 0.02
Km_0 = 1E4 # microgram glucose per liter

Km = (Km_0 * 1E-6) / (180.16)
phi_O = 0.35
phiX_range = np.linspace(0, 0.2, 20)
phi_R = 0.3
phiP_range = 1 - phiX_range  - phi_R - phi_O
M0 = 0.1 * OD_CONV
M_P = phiP_range * M0
M_R = phi_R * M0
m_AA = 0.001 * M0
m_N = 0.010 * 6.022E23 * 1E-3 
nu = 5
omega = 0.377 # * OD_CONV


dfs = []
time_range = np.linspace(0, 5, 300)
for i, phiP in enumerate(tqdm.tqdm(phiP_range)):
    params = [M0, M_R, M_P[i], m_AA, m_N] 
    args = (gamma_max, nu, omega, phi_R, phiP, Kd, Km)

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
    _df['phi_X'] = phiX_range[i]
    dfs.append(_df)
df = pd.concat(dfs)


#%%
# Define the colors
cmap = sns.color_palette(f"rocket", n_colors=len(phiX_range))

fig, ax = plt.subplots(3, 2, figsize=(6, 5))
# ax[0, 0].axis('off')
# ax[0, 1].set_yscale('log')
for a in ax.ravel():
    a.set_xlabel(r'time $\times \gamma_{max}$')

# Add other lables
ax[0,0].set(title='total biomass',
            ylabel=r'$M_t /  M_0$')
ax[1,0].set(title='translational efficiency',
            ylabel=r'$\gamma_t /  \gamma_{max}$')
ax[1,1].set(title=r'nutritional efficiency, $\nu$',
            ylabel=r'$\nu_t /  \nu_{max}$')
ax[2,0].set(title='charged-tRNA concentration',
            ylabel=r'$m_{AA} / M$')
ax[0,1].set(title='nutrient concentration',
            ylabel=r'$c_{N,t} / c_{N,0}$')

count = 0
for g, d in df.groupby('phi_X'):
    ax[0, 0].plot(d['time'], d['rel_biomass'], '-', lw=1, color=cmap[count])
    ax[1, 0].plot(d['time'], d['gamma'], '-', lw=1, color=cmap[count])
    ax[1, 1].plot(d['time'], d['nu'], '-', lw=1, color=cmap[count])
    ax[2, 0].plot(d['time'], d['c_aa'], lw=1, color=cmap[count])
    ax[0, 1].plot(d['time'], d['c_n'], lw=1, color=cmap[count])
    count += 1
ax[-1, -1].axis('off')
plt.subplots_adjust(wspace=0.2, hspace=0.1)

plt.tight_layout()
plt.savefig('../../../figures/presentations/dynamics_useless_expression.pdf')
# if ALL_NU == False:
    # plt.savefig('../../../figures/presentations/dynamics_plots_single_nu.pdf')
# else:
    # plt.savefig('../../../figures/presentations/dynamics_plots_all_nu.pdf')
# %%
# Look only at the steady state time regime

fig, ax = plt.subplots(1, 3, figsize=(6, 2))
# ax[0, 0].axis('off')
# ax[0, 1].set_yscale('log')
for a in ax:
    a.set_xlabel(r'time $\times \gamma_{max}$')

# Add other lables
ax[0].set(title='total biomass',
            ylabel=r'$M_t /  M_0$')
ax[1].set(title=r'nutritional efficiency, $\nu$',
            ylabel=r'$\nu_t /  \nu_{max}$')
ax[2].set(title='charged-tRNA concentration',
            ylabel=r'$c_{AA} / M$')
ax[0].set_yscale('log')
count = 0
for g, d in df.groupby('nu_max'):
    ax[0].plot(d['time'], d['rel_biomass'], '-', lw=1, color=cmap[count])
    ax[1].plot(d['time'], d['nu'], '-', lw=1, color=cmap[count])
    ax[2].plot(d['time'], d['c_aa'], lw=1, color=cmap[count])
    count += 1  
for a in ax:
    a.set_xlim([5, 10])

ax[0].set_ylim([1, 15])
ax[1].set_ylim([0.8, 1])
ax[1].set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
plt.tight_layout()
plt.savefig('../../../figures/presentations/dynamics_plots_steady_state.pdf')
# %%
