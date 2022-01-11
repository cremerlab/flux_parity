#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import growth.model
import growth.viz 
import growth.integrate
import scipy.stats
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# # Load constants
# gamma_max = const['gamma_max']
# phi_O = const['phi_O']
# tau = const['tau']
# Kd_TAA = const['Kd_TAA']
# Kd_TAA_star = const['Kd_TAA_star']
# kappa_max = const['kappa_max']
# nu_max = 3 

# # Equilibrate the model
# args = {'gamma_max': gamma_max,
#         'nu_max': nu_max,
#         'tau': tau,
#         'phi_O': phi_O,
#         'Kd_TAA': Kd_TAA,
#         'Kd_TAA_star': Kd_TAA_star,
#         'kappa_max': kappa_max,
#         'phi_O': phi_O}

# out = growth.integrate.equilibrate_FPM(args)
# phiRb = out[1] / out[0]
# ratio = out[-1] / out[-2]
# gamma = gamma_max * out[-1] / (out[-1] + Kd_TAA_star)



#%%
n_ribos = int(1E8)
bgal_len = 1020 # Length of LacZ in AA    
mu = 12
std = 3 # standard deviation of the distribution 
time_range = np.arange(0,  300, 1)
# v_tl = np.random.normal(mu, std, size=(len(time_range), n_ribos)

tot_bgal = np.zeros_like(time_range)

# Scenario I: Assign each ribosome a translation rate
v_tl = np.abs(np.random.normal(mu, std, size=n_ribos))

# Set the time range 
synthesis = np.zeros((len(time_range), len(v_tl)))
start_times = np.random.normal(10, 2, size=len(v_tl))
for i, t in enumerate(tqdm.tqdm(time_range)):
    started = t >= start_times
    synthesis[i, :] = v_tl * t * started

# Assume we are at sta
bgal_per_ribo = np.floor(synthesis / bgal_len)
tot_bgal = np.sum(bgal_per_ribo, axis=1)

#%%
# Set the sub sampling
step = 15 # in s
sub_time = time_range[::15]
miller_u = 5E-6#np.random.normal(5E-5, 5E-6, len(tot_bgal[::15])) # miller units per lac, chosen to make units comparable to Dai Fig 1.
sub_bgal = tot_bgal[::15] * miller_u

# Add background signal and some noise
sub_bgal += 5 #np.random.normal(10, 0.1, size=len(sub_bgal))

# As per the SI of dai, compute the E0 using the first three point 
E0 = sub_bgal[:3].mean()
Et = sub_bgal - E0
output = np.sqrt(Et - E0)

# Do a linear fit to this region 
n_points = len(output[output > 0])
ind = len(output) - n_points
popt = scipy.stats.linregress(sub_time[ind:]-10, output[output > 0])
slope = popt[0]
yintercept = popt[1]
t_first = -yintercept/slope

# Given t_first, compute the elongation rate
elongation_rate = bgal_len / t_first

fig, ax = plt.subplots(1, 3, figsize=(8, 2.5))
ax[0].plot(sub_time, sub_bgal, 'o', label='simulated data')
ax[0].set(xlabel='time from induction [s]', ylabel='approximate Miller units')
ax[1].plot(sub_time, output, 'o')
ax[1].set(xlabel='time from induction [s]', 
          ylabel=r'$\sqrt{\beta-gal(t) - \langle\beta-gal(t \leq 45 sec)\rangle}$',
          xlim=[0, 300])
ax[2].set(xlabel='single-ribosome translation rate', ylabel='probability')
# Plot the fit
fit_time = np.linspace(t_first, 300, 100)
ax[1].plot(fit_time,  fit_time * slope + yintercept, 'k-', label='linear regression')
ax[1].legend()
_ = ax[2].hist(v_tl, edgecolor='k', bins=100, alpha=0.4, 
            color=colors['primary_blue'], density=True,
            label='seed distribution')
ylim = ax[2].get_ylim()[1]
ax[2].vlines(mu, 0, ylim, lw=1, color=colors['primary_black'], label='true mean')
ax[2].vlines(elongation_rate, 0, ylim, lw=1, color=colors['primary_red'], label='inferred mean')
ax[2].legend(loc='upper left')
# plt.savefig('./elongation_rate_measurement_simulation.pdf', bbox_inches='tight')
# %%

# %%
