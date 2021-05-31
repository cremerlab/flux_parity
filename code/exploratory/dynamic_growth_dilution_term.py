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
Km_0 = 1E5 # microgram glucose per liter

Km_0 = 1E-10 #(Km_0 * 1E-6) / (180.16)
phi_O = 0.35
phi_R = 0.3
phi_P = 1 - phi_R - phi_O
M0 = 0.1 * OD_CONV
M_P = phi_P * M0
M_R = phi_R * M0
c_AA = 0.001 # * M0
m_N = 1E50 #1E8 * 6.022E23 * 1E-3 
omega = 0.377 # * OD_CONV


dfs = []
nu_range = np.linspace(0.005, 10, 8)
time_range = np.linspace(0, 10, 300)
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    for dil in [True, False]:  
        params = [M0, M_R, M_P, c_AA, m_N] 
        args = (gamma_max, nu, omega, phi_R, phi_P, Kd, Km, dil)
        out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator, 
                                    params, time_range, args=args)
        _df = pd.DataFrame(out, columns=['biomass', 'ribos', 'metabs', 'c_aa', 'm_n'])
        _df['c_n'] = _df['m_n'].values / (6.022E23 * 1E-3)
        _df['m_n'] = _df['m_n'] / m_N
        _df['rel_biomass'] = _df['biomass'].values / M0
        _df['time'] = time_range * gamma_max
        _df['gamma'] =  (_df['c_aa'].values / (_df['c_aa'].values + Kd)) 
        _df['nu'] =  (_df['c_n'].values / (_df['c_n'].values + Km))
        _df['nu_max'] = nu
        _df['dilution_ignored'] = dil
        dfs.append(_df)
df = pd.concat(dfs)
#%%
# Define the colors
cmap = sns.color_palette(f"mako", n_colors=len(nu_range))

fig, ax = plt.subplots(3, 2, figsize=(3.5, 5))
ax[0, 0].axis('off')
ax[0, 1].set_yscale('log')
for a in ax.ravel():
    a.set_xlabel(r'time $\times \gamma_{max}$')

# Add other lables
ax[0,1].set(title='total biomass',
            ylabel=r'$M_t /  M_0$')
ax[1,0].set(title='translational efficiency',
            ylabel=r'$\gamma_t /  \gamma_{max}$')
ax[1,1].set(title=r'nutritional efficiency, $\nu$',
            ylabel=r'$\nu_t /  \nu_{max}$')
ax[2,0].set(title='charged-tRNA concentration',
            ylabel=r'$c_{AA}$')
ax[2,1].set(title='nutrient concentration',
            ylabel=r'$c_{N,t} / c_{N,0}$')

# Add panel lablels
fig.text(0.01, 0.99, '[A]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.99, '[B]', fontsize=8, fontweight='bold')
fig.text(0.01, 0.645, '[C]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.645, '[D]', fontsize=8, fontweight='bold')
fig.text(0.01, 0.32, '[E]', fontsize=8, fontweight='bold')
fig.text(0.5, 0.32, '[F]', fontsize=8, fontweight='bold')

_colors = {n:cmap[i] for i, n in enumerate(df['nu_max'].unique())}
for g, d in df.groupby(['nu_max', 'dilution_ignored']):
    if g[1] == True:
        ls ='--'
    else:
        ls = '-'
    ax[0, 1].plot(d['time'], d['rel_biomass'], ls, color=_colors[g[0]])
    ax[1, 0].plot(d['time'], d['gamma'], ls,  color=_colors[g[0]])
    ax[1, 1].plot(d['time'], d['nu'], ls, color=_colors[g[0]])
    ax[2, 0].plot(d['time'], d['c_aa'], ls, color=_colors[g[0]])
    ax[2, 0].set_yscale('log')
    ax[2, 1].plot(d['time'], d['c_n'], ls, color=_colors[g[0]])


plt.subplots_adjust(wspace=0.2, hspace=0.2)
plt.tight_layout()
plt.savefig('../../figures/dilution_term_ingored.pdf')

# %%
fig, ax = plt.subplots(1, 2, sharey=True)
for g, d in df.groupby(['nu_max', 'dilution_ignored']):
    _diff = np.diff(d['c_aa'].values)
    if g[1] == True:
        a = ax[1]
    else: 
        a = ax[0]
    a.plot(d['time'].values[:-1], _diff, '-', color=_colors[g[0]],
            lw=1)
ax[0].set_title('dilution term kept')
ax[1].set_title('dilution term ignored')
ax[0].set_ylabel(r'$d c_{AA} / dt$')
for a in ax:
    a.set_xlabel('time hr')
    a.set_xlim([1,18])
    # a.set_ylim([0, 0.005])
    # a.set_yscale('log')
# %%

# %%

# %%

# %%

# %%
