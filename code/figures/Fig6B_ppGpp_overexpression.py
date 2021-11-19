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
mapper = growth.viz.load_markercolors()

# Load the data   
data = pd.read_csv('../../data/Scott2010_lacZ_overexpression.csv')
cmap  = sns.color_palette('bone', n_colors=3)
# marker_mapper = {'β-lactamase': 's',
#                  'EF-Tu': 'd',
#                  'β-galactosidase': 'o'}

# color_mapper = {'Dong et al., 1995': cmap[2],
#                 'Bentley et al., 1990': cmap[1],
#                 'Scott et al., 2010': cmap[0]}

dt = 0.001
time_range = np.arange(0, 150, dt)
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_max =  [9, 2.8, 1.8]
phi_O = 0.55 # As reported in Hui et al. 2015 (Table S7)

phiX_range = np.linspace(0, 0.35)

M0 = 1
M_Rb = 0.01 * M0
M_Mb = (1 -  phi_O - phiX_range - 0.01) * M0
TAA = 1E-5
TAA_star = 5E-6
df = pd.DataFrame([])
for j, n in enumerate(tqdm.tqdm(nu_max)):
    for i, x in enumerate(tqdm.tqdm(phiX_range)):
        params = [M0, M_Rb, M_Mb[i], TAA, TAA_star]
        args = (gamma_max, n, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O + x)
        out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                        params, time_range, args=args)
        out = out[-1]
        gr = gamma_max * (out[1] / out[0]) * out[-1] / (out[-1] + Kd_TAA_star)
        if i == 0:
            gr_init = gr
        df = df.append({'phi_X': x, 'growth_rate_hr':gr,  'nu':n}, ignore_index=True)

#%%
fig, ax = plt.subplots(1, 1, figsize=(1.85, 1.85))
ax.set_xlabel('allocation towards LacZ\n$\phi_X$')
ax.set_ylabel(r'$\lambda$ [hr$^{-1}$]' + '\ngrowth rate')
cmap = [colors['dark_red'], colors['primary_red'], colors['light_red']]

counter = 0
for g, d in data.groupby(['medium_id']):
    ax.plot(d['phi_X'].values / 100, d['growth_rate_hr'], 'o', linestyle='none', 
             ms=4,  label=g, zorder=1000, markeredgewidth=0.25, alpha=0.75,
             color=cmap[counter], markeredgecolor='k')
    counter += 1 

ax.set_ylim([0, 2])
ax.set_yticks([0, 0.5, 1, 1.5, 2])

cmap.reverse()
counter = 0
for g, d in df.groupby(['nu']):
    ax.plot(d['phi_X'], d['growth_rate_hr'], '--', lw=1, color=cmap[counter])
    counter += 1
plt.savefig('../../figures/Fig6B_lacZ_overexpression.pdf', bbox_inches='tight')


# %%
