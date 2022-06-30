#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.model
import growth.integrate
import growth.viz
import seaborn as sns
import tqdm
const = growth.model.load_constants()
colors,  palette = growth.viz.matplotlib_style()
mapper = growth.viz.load_markercolors()
#%%
# Load the Dai data
ribo_chlor_data = pd.read_csv('../data/main_figure_data/Fig5A_Dai2016_chloramphenicol_ribosome_content.csv')
elong_chlor_data = pd.read_csv('../data/main_figure_data/Fig5A_Dai2016_chloramphenicol_elongation_rates.csv')

# Load the comparison data

# Define constant parameters
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']

# Define parameter ranges for non-chlor case
nu_range = np.linspace(0.1, 20, 200)

# Compute the non-chlor case
nochlor_df = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    # Define the arguments
    args = {'gamma_max':gamma_max,
            'nu_max': nu,
            'tau': tau,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O}

    # Equilibrate the model and print out progress
    out = growth.integrate.equilibrate_FPM(args, tol=2, max_iter=1) 

    # Assemble the dataframe
    ratio = out[-1] / out[-2]
    phiRb = (1 - phi_O) * (ratio / (ratio + tau))
    gamma = gamma_max * (out[-1] / (out[-1] + Kd_TAA_star))
    nochlor_df = nochlor_df.append({'MRb_M': out[1]/out[0],
                                    'phiRb': phiRb,
                                    'gamma': gamma,
                                    'v_tl': gamma * 7459 / 3600,
                                    'nu': nu,
                                    'lam': gamma * phiRb}, 
                                    ignore_index=True)

#%%
# Estimate the best nu for each growth medium
nu_mapper = {}
for g, d in ribo_chlor_data[ribo_chlor_data['chlor_conc_uM']==0].groupby(['medium']):
    phiRb = d['mass_fraction'].values[0] 
    lam = d['growth_rate_hr'].values[0]
    estimated_nu = growth.integrate.estimate_nu_FPM(phiRb, lam, const, phi_O, 
                                                    verbose=True, nu_buffer=1,
                                                    tol=2)

    nu_mapper[g] = estimated_nu

#%%

# Using the estimated nus, perform the integration
chlor_range = np.linspace(0, 12.5, 10) * 1E-6
# Compute the non-chlor case
chlor_df = pd.DataFrame([])
dt = 0.0001
for medium, nu in nu_mapper.items():
    # Define the arguments

    for i, c in enumerate(chlor_range):
        args = {'gamma_max':gamma_max,
            'nu_max': nu,
            'tau': tau,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O,
            'antibiotic': {
                'c_drug': c,
                'Kd_drug': 5E-10,
            }}

        # Equilibrate the model and print out progress for experimentally relevant
        # time range (default 200hrs)
        out = growth.integrate.equilibrate_FPM(args, tol=2, max_iter=1, t_return=2, dt=dt) 
        _out = out[-1] 
        gr = np.log(out[-1][0]/out[-2][0]) / dt

        # Assemble the dataframe
        ratio = _out[-1] / _out[-2]
        phiRb = (1 - phi_O) * (ratio / (ratio + tau))
        gamma = gamma_max * (_out[-1] / (_out[-1] + Kd_TAA_star))
        chlor_df = chlor_df.append({'MRb_M': _out[1]/_out[0],
                                    'phiRb': phiRb,
                                    'gamma': gamma,
                                    'v_tl': gamma * 7459 / 3600,
                                    'nu': nu,
                                    'lam': gr,
                                    'medium': medium,
                                    'chlor_conc': c}, 
                                    ignore_index=True)

#%%
# Instantiate canvases
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].axis('off')

# Format axes
ax[1].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6)
ax[1].set_ylabel('$\phi_{Rb}$\nallocation towards ribosomes', fontsize=6)
ax[1].set(xlim=[0, 2.5], ylim=[0, 0.4])
ax[2].set_xlabel('growth rate [hr$^{-1}$]\n$\lambda$', fontsize=6),
ax[2].set_ylabel('$v_{tl}$\ntranslation speed [AA / s]',fontsize=6) 
ax[2].set(xlim=[0, 2.5], ylim=[3, 22])
ax[1].set_yticks([0, 0.1, 0.2, 0.3, 0.4])
ax[2].set_yticks([5, 10, 15, 20])
# ax[0].set_xlabel(fontsize=6)

# Manually define color series
cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=6) 
cmap = [cmap[2], cmap[1], cmap[3], cmap[5], cmap[4], cmap[0]]
# Plot the chlor data
counter = 0
for g, d in ribo_chlor_data.groupby(['medium']):
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', ms=4,
        alpha=0.75, markeredgecolor='k', markeredgewidth=0.5, color=cmap[counter])
    counter += 1

# Plot the chlor data
counter = 0
for g, d in elong_chlor_data.groupby(['medium']):
    ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o', ms=4,
        alpha=0.75, markeredgecolor='k', markeredgewidth=0.5,
        color=cmap[counter])
    counter += 1

# Plot the nochlor theory
ax[1].plot(nochlor_df['lam'], nochlor_df['phiRb'], '--', lw=1, 
                                                  color=colors['primary_black'])
ax[2].plot(nochlor_df['lam'], nochlor_df['v_tl'], '--', lw=1, 
                                                  color=colors['primary_black'])

# Plot the chlor theory
counter = 0
for g, d in chlor_df.groupby(['medium']):
    ax[1].plot(d['lam'], d['MRb_M'], '-', color=cmap[counter], lw=1)
    ax[2].plot(d['lam'], d['v_tl'], '-', color=cmap[counter], lw=1)
    counter +=1
plt.tight_layout()
plt.savefig('../figures/main_text/plots/Fig3_chloramphenicol.pdf', bbox_inches='tight')
# %%
