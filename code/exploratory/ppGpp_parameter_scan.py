#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate
import growth.model
import growth.viz
import multiprocessing as mp
from joblib import Parallel, delayed
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
#%%

# Define constant params
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
phi_O = 0.55
nu_max = 4
kappa_max = const['kappa_max']

# Compute the optimium
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)

# Define parameter ranges
tau_range = [1]
Kd_TAA_range = np.logspace(-5, -2, 40)
Kd_TAA_star_range = np.logspace(-5, -2, 40)

df = pd.DataFrame([])
# eqs = Parallel(n_jobs=mp.cpu_count())(delayed(growth.model.equilibrate_ppGpp)(
#         {'gamma_max':gamma_max, 'nu_max':nu_max,
#          'tau': t, 'Kd_TAA': kd, 'Kd_TAA_star': kd_star,
#          'kappa_max':kappa_max, 'phi_O':phi_O}) for kd_star in Kd_TAA_star_range for kd in Kd_TAA_range for t in tau_range)

#%%
out = np.zeros((len(Kd_TAA_range), len(Kd_TAA_star_range)))
for i, tau in enumerate(tqdm.tqdm(tau_range)):
    for j, kd in enumerate(tqdm.tqdm(Kd_TAA_range)):
        for k, kd_star in enumerate(Kd_TAA_star_range):
            args = {'gamma_max':gamma_max, 
                    'nu_max':  nu_max, 
                    'tau': tau, 
                    'Kd_TAA': kd, 
                    'Kd_TAA_star': kd_star, 
                    'kappa_max': kappa_max, 
                    'phi_O': phi_O}
            equil = growth.model.equilibrate_ppGpp(args)
            out[j, k] = 1 - np.sqrt(((equil[1] / equil[0]) -  opt_phiRb)**2)
            # df = df.append({'fp_phiRb': equil[1]/equil[0], 
            #                 'opt_phiRb':opt_phiRb,
            #                 'tau': tau,
            #                 'Kd_TAA': kd,
            #                 'Kd_TAA_star': kd_star}, 
            #                 ignore_index=True)

# %%
# df['diff'] = np.sqrt((df['fp_phiRb'].values - df['opt_phiRb'].values)**2)

# %%
fig, ax = plt.subplots(1, 1)
_im = ax.matshow(np.log(1 - out), cmap='mako', origin='lower', interpolation='none')
# plt.colorbar()
ax.plot(np.arange(0, 40,1), 'r--', lw=1)
ax.grid(False)
ax.set_xticks([0, 10, 20, 30, 40])
ax.set_yticks([0, 10, 20, 30, 40])
ax.set_xticklabels(['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$'])
ax.set_yticklabels(['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$'])
ax.set_xlabel('uncharged-tRNA $K_D$')
ax.set_ylabel('charged-tRNA $K_D$')
fig.colorbar(_im, ax=ax, label='log\u2081\u2080 difference from optimal allocation')
plt.savefig('../../figures/FPO_Kd_parameter_scan.pdf', bbox_inches='tight')
# %%
