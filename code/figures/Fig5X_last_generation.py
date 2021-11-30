#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
import scipy.integrate
import growth.model 
import growth.integrate
import growth.viz
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
peak_data = pd.read_csv('../../data/Bren2013_FigS3A.csv')
od_data = pd.read_csv('../../data/Bren2013_FigS3B.csv')

#%%
# Define constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
Kd_cnt = 5E-6
Y = 1E23#2.5E23#const['Y']
tau = const['tau']
phi_O = 0.55
dt = 0.0001
c_nt = [0.00022, 0.00044, 0.0011]
#%% Estimate the metabolic rate given a growth rate of 1 per hr and phiR of approx 0.1
nu = growth.integrate.estimate_nu_FPM(0.1, 0.9, const, phi_O, nu_buffer=3, tol=2,
                                      verbose=True)
#%%
# nu = 2.5
# Equilibrate
args = {'gamma_max':gamma_max,
        'nu_max': nu,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max': kappa_max,
        'tau': tau,
        'phi_O':phi_O}

out = growth.integrate.equilibrate_FPM(args) 
phiRb = out[1] / out[0]
T_AA = out[-2]
T_AA_star = out[-1]

# Given equilibration, set up actual integration
M0 = 1E18
M_Rb= phiRb * M0
M_Mb = (1 - phiRb - phi_O) * M0
args = {'gamma_max':gamma_max,
            'nu_max':nu,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'tau': tau,
            'phi_O': phi_O,
            'nutrients': {'Kd_cnt': Kd_cnt,
                           'Y': Y}}

growth_time = np.arange(0, 12, dt)
dfs = []
for i, c in enumerate(c_nt):
    init_params = [M0, M_Rb, M_Mb, c, T_AA, T_AA_star]
    growth_cycle = scipy.integrate.odeint(growth.model.self_replicator_FPM,
                                          init_params, growth_time, args=(args,))
    df = pd.DataFrame(growth_cycle, columns=['M', 'M_Rb', 'M_Mb', 'c_nt', 'TAA', 'TAA_star'])
    df['balance'] = df['TAA_star'].values / df['TAA'].values
    df['phiRb'] = (1 - phi_O) * df['balance'].values / (df['balance'].values + tau)
    df['MMb_M'] = df['M_Mb'].values / df['M']
    df['MRb_M'] = df['M_Rb'].values / df['M']
    df['phiMb'] = 1 - df['phiRb'].values - phi_O
    df['gamma'] = gamma_max * (df['TAA_star'].values / (df['TAA_star'].values + Kd_TAA_star))
    df['c_nt_max'] = c
    df['nu'] =  nu * (df['TAA'].values / (df['TAA'].values + Kd_TAA)) * (df['c_nt'].values / (df['c_nt'].values + Kd_cnt))
    df['time'] = growth_time
    dfs.append(df)
df = pd.concat(dfs, sort=False)
df['OD'] = df['M'].values / 1E18 

# Instantiate the figure canvas
fig, ax = plt.subplots(1, 2, figsize=(4.5, 1.75), sharex=True)

# Format axes
ax[0].set(ylabel='optical density [a.u.]',
          xlabel='time [hr]',
          ylim=[1E-3, 1E-1],
          xlim=[2, 7],
          yscale='log')
ax[1].set(ylabel='change in PtsG\npromoter activity',
          xlabel='time [hr]',
          yticks=[],
          xlim=[2, 7])

# Set up the second y axis
ax2 = ax[1].twinx()
ax2.set_ylabel('change in metabolic\nprotein expression', color=colors['red'])
ax2.set_yticks([])

# Plot the data
cmap = sns.color_palette(f"light:{colors['black']}_r", n_colors=4)
counter = 0 
for g, d in peak_data.groupby(['medium']):
    if g != '11 mM Glucose':
        ax[1].plot(d['time_hr'], d['promoter_activity'], '-', lw=1.5, color=cmap[counter],
        alpha=0.75, label=g)
    counter += 1

counter = 0
for g, d in od_data.groupby(['medium']):
    ax[0].plot(d['time_hr'].values[::3], d['od_600nm'].values[::3], 'o', ms=4, 
               color=cmap[counter], alpha=0.5, markeredgecolor='k', 
               markeredgewidth=0.5)
    counter += 1

# Plot the theory
cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=4)
counter = 0
for g, d in df.groupby(['c_nt_max']):
    deriv = np.diff(d['MMb_M'].values)

    # Find the maximum point of the derivative
    ax2.plot(d['time'].values[:-1],  deriv, '--', label=f'{g:0.4f}',
                color = cmap[counter], lw=1, zorder=1000)
    ax[0].plot(d['time'], d['OD']/2100, '--', label=g,
                color = cmap[counter], lw=1)
    counter += 1

# ax[0].set_ylim([1E-3, 1E-1])
# ax[0].set_yticks([1E-3, 1E-2,  1E-1])
plt.tight_layout()
plt.savefig('../../figures/Fig5X_final_generation.pdf', bbox_inches='tight')

# %%
a
# %%
