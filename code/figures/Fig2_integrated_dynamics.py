#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import growth.viz
import growth.model
import scipy.integrate
colors, _ =  growth.viz.matplotlib_style()

# Define the volume constants
VOL = 1E-3 # in L

# Define the efficiencies
gamma_max = 9.65
nu_max = np.linspace(0.1, 5, 25)
palette = sns.color_palette('crest', n_colors=len(nu_max))

# Define the yield coefficient 
nutrient_mw = 180 * (6.022E23 / 110) # in AA mass units per mol 
omega = 0.3 * VOL * nutrient_mw # in units of Vol * AA mass / mol

# Define the dissociation constants  
Kd_cN = 5E-4  # in M
Kd_cAA = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23) # in abundance units 

# Define the allocation parameters
phi_R = 0.3
phi_P = 0.35

# Define other useful constants
OD_CONV = 1.5E17 

# Define starting masses
M0 = 0.1 * OD_CONV
MR_0 = phi_R * M0
MP_0 = phi_P * M0

# Define starting concentrations
cAA_0 = 1E-3 # in abundance units
cN_0 = 0.010

# Set up the time range of the integration
time_range = np.linspace(0, 20, 300)

# Perform the integration for each value of nu max
dfs = []
for i, nu in enumerate(nu_max):

    # Pack parameters and arguments
    params = [M0, MR_0, MP_0, cAA_0, cN_0] 
    args = (gamma_max, nu, omega, phi_R, phi_P, Kd_cAA, Kd_cN)

    # Perform the integration
    out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator, 
                                params, time_range, args=args)

    # Pack the integration output into a tidy dataframe
    _df = pd.DataFrame(out, columns=['biomass', 'ribos', 'metabs', 'c_AA', 'c_N'])
    _df['rel_biomass'] = _df['biomass'].values / M0
    _df['time'] = time_range
    _df['gamma'] = gamma_max * (_df['c_AA'].values / (_df['c_AA'].values + Kd_cAA))
    _df['time'] = time_range * gamma_max
    _df['nu_max'] = nu 
    dfs.append(_df)
df = pd.concat(dfs)

# Instantiate figure and label/format axes
fig, ax = plt.subplots(1, 3, figsize=(5, 1.8), sharex=True)
for a in ax:
    a.xaxis.set_tick_params(labelsize=6)
    a.yaxis.set_tick_params(labelsize=6)
    a.set_xlabel(r'time $\times \gamma_{max}$', fontsize=8, color=colors['primary_black'])
ax[0].set_ylabel(r'$M(t)\, / \, M(t=0)$', fontsize=8, color=colors['primary_black'])
ax[1].set_ylabel(r'$c_{AA}(t)\, /\,  K_D^{c_{AA}}$', fontsize=8, color=colors['primary_black'])
ax[2].set_ylabel(r'$c_N(t)\, /\, K_D^{c_{N}}$', fontsize=8, color=colors['primary_black'])
ax[0].set_yscale('log')
# ax[2].set_ylim([0, 1.1])
count = 0
for g, d in df.groupby('nu_max'):
    ax[0].plot(d['time'], d['rel_biomass'], '-', lw=0.5, color=palette[count])
    ax[1].plot(d['time'], d['c_AA'] / Kd_cAA, '-', lw=0.5, color=palette[count])
    ax[2].plot(d['time'], d['c_N']  / Kd_cN, '-', lw=0.5, color=palette[count])
    count += 1
plt.tight_layout()
# plt.savefig('../../figures/Fig2_dynamics_plots.pdf', bbox_inches='tight')

# %%

# %%
