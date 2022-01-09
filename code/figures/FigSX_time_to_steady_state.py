#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tqdm
import growth.viz 
import growth.model
import scipy.integrate
import growth.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

#%%
# Load invariant constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']

# Estimate nu max for an initial growth rate
lam_init = 2
# estimated_nu = growth.integrate.estimate_nu_FPM(0.25, 
                                                # lam_init, 
                                                # const, 
                                                # phi_O, 
                                                # verbose=True, 
                                                # guess = 6)
#%%
# Empirically determined nu_max for fast and slow growth
fast = 11.8
fat_lam = 2
slow = np.linspace(0.2, 11.8, 100)
slow_lam = 0.36
dt = 0.001
n_gen = 10

# Set the initial params
args = {'gamma_max': gamma_max,
        'nu_max': fast,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max': kappa_max,
        'tau': tau,
        'phi_O': const['phi_O']}
out = growth.integrate.equilibrate_FPM(args)
TAA = out[-2]
TAA_star = out[-1]
ratio = TAA_star / TAA
phiRb = (1 - phi_O) * (ratio / (ratio + tau))
lam = gamma_max * phiRb * (TAA_star / (TAA_star + TAA))

#%%
dfs = []
for i, s in enumerate(tqdm.tqdm(slow)):
    if  s < 0.1:
        harvest_time = 24
    elif (s < 0.2) & (s > 0.1):
        harvest_time = 14 
    else:
        harvest_time = 14
    time_range = np.arange(0, harvest_time, dt)
    args = {'gamma_max': gamma_max,
            'nu_max': s,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'tau': tau,
            'phi_O': const['phi_O']}
    out = growth.integrate.equilibrate_FPM(args)
    # slow_RNA_prot
    slow_TAA = out[-2]
    slow_TAA_star = out[-1]
    slow_ratio = TAA_star / TAA
    slow_phiRb = (1 - phi_O) * (slow_ratio / (slow_ratio + tau))
    slow_lam = gamma_max * slow_phiRb * (slow_TAA_star / (slow_TAA_star + slow_TAA))


    # Set the params for the diluted culture
    M0 = 0.001 * const['OD_conv']
    params = [M0, phiRb * M0, (1 - phi_O - phiRb) * M0, TAA, TAA_star]
    preculture = scipy.integrate.odeint(growth.model.self_replicator_FPM, params, time_range, (args,)) 

    df = pd.DataFrame(preculture, columns=['M', 'M_Rb', 'M_Mb', 'TAA', 'TAA_star'])
    df['time'] = time_range
    df['n_gen'] = time_range * s
    df['nu_max'] = s
    df['growth_rate'] = slow_lam
    dfs.append(df)


# %%
df = pd.concat(dfs, sort=False)
df['MRb_M'] = df['M_Rb'].values / df['M'].values
df['RNA_prot'] = df['MRb_M'] / 0.4558
df['ratio'] = df['TAA_star'].values / df['TAA'].values
df['phi_Rb'] = (1 - phi_O) * (df['ratio'].values / (df['ratio'].values + tau))
df['approx_OD'] = df['M'] / const['OD_conv']

#%%
harvest = pd.DataFrame([])
for g, d in df.groupby(['nu_max']):
    harvest = harvest.append({'nu_max':g,
                              'phi_Rb':d['phi_Rb'].values[-1],
                              'MRb_M': d['MRb_M'].values[-1],
                              'lam': d['growth_rate'].values[-1]},
                              ignore_index=True) 
#%%
fig, ax = plt.subplots(1,1)
ax.plot(harvest['lam'], harvest['MRb_M'], 'k-', lw=2)
ax.plot(harvest['lam'], harvest['phi_Rb'], '--', color=colors['primary_red'], lw=2)
ax.set_ylim([0, 0.30])
# %%
fig, ax = plt.subplots(2, 2, figsize=(6,6), sharex=True)
ax[0,0].plot(df['time'], df['approx_OD'], 'k-')
ax[0,0].set(yscale='log', ylabel='approximate optical density')
ax[0, 1].plot(df['time'], df['MRb_M'], 'k-')
ax[0, 1].set(xlabel='time [hr]', ylabel='RNA / protein') #,  ylim=[0, 0.15])

# %%
