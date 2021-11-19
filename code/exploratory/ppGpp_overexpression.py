#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.viz 
import growth.model
import tqdm
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Load the data   
# data = pd.read_csv('../../data/Scott2010_lacZ_overexpression.csv')
data = pd.read_csv()
markers = ['o', 's', 'X']


dt = 0.001
time_range = np.arange(0, 150, dt)

gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_max =  [9, 2.8, 1.8]
phi_O = 0.55 # As reported in Scott 2010

phiX_range = np.linspace(0, 0.35)

M0 = 1
M_Rb = 0.01 * M0
M_Mb = (1 -  phi_O - phiX_range - 0.01) * M0
TAA = 1E-5
TAA_star = 5E-6
df = pd.DataFrame([])
for j,  nu in enumerate(nu_max):
    for i, x in enumerate(tqdm.tqdm(phiX_range)):
        params = [M0, M_Rb, M_Mb[i], TAA, TAA_star]
        args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O + x)
        out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                        params, time_range, args=args)
        out = out[-1]
        gr = gamma_max * (out[1] / out[0]) * out[-1] / (out[-1] + Kd_TAA_star)
        df = df.append({'phi_X': x, 'growth_rate_hr':gr, 'nu':nu}, ignore_index=True)


fig, ax = plt.subplots(1, 1, figsize=(2, 2))
ax.set_xlabel('allocation towards LacZ\n$\phi_X$')
ax.set_ylabel(r'$\lambda$ [hr$^{-1}$]' + '\nrelative growth rate')

counter = 0
labels = ['RDM + glucose', 'cAA + glucose', 'M63 + glucose']
for g, d in data.groupby(['medium_id']):
    ax.plot(d['phi_X'].values / 100, d['growth_rate_hr'], linestyle='none', 
            markerfacecolor=colors['primary_black'],
             ms=4, marker=markers[counter], label=labels[counter],
            zorder=1000, markeredgewidth=0.25, alpha=0.75)
    counter += 1
ax.set_ylim([0, 2])
ax.legend()

counter = 0
# Reset colors to match indices
# _palette = [colors['primary_green'], colors['primary_blue'], colors['primary_black']]
# palette[:-1].reverse()
for g, d in df.groupby(['nu']):
    ax.plot(d['phi_X'], d['growth_rate_hr'], '--', color=colors['primary_black'], lw=1)
    counter += 1
ax.set_yticks([0, 0.5, 1, 1.5, 2])
plt.savefig('../../figures/lacZ_overexpression.pdf', bbox_inches='tight')

# %%
a = 1

# %%
