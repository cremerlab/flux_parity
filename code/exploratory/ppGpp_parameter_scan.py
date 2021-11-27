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
tau = 1
kappa_max = const['kappa_max']
Kd_TAA = 1E-5
Kd_TAA_star = 1E-5
# Compute the optimium
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)

# Define parameter ranges
resolution = 40
tau_range = np.logspace(-3, 1, resolution)
kappa_range = np.logspace(-5, 0, resolution)
Kd_TAA_range = np.logspace(-5, -2, resolution)
Kd_TAA_star_range = np.logspace(-5, -2, resolution)

#%%
Kd_sweep_out = np.zeros((len(Kd_TAA_range), len(Kd_TAA_star_range)))
for i, kd in enumerate(tqdm.tqdm(Kd_TAA_range)):
    for j, kd_star in enumerate(Kd_TAA_star_range):
        args = {'gamma_max':gamma_max, 
                'nu_max':  nu_max, 
                'tau': tau, 
                'Kd_TAA': kd, 
                'Kd_TAA_star': kd_star, 
                'kappa_max': kappa_max, 
                'phi_O': phi_O}
        equil = growth.model.equilibrate_ppGpp(args)
        Kd_sweep_out[i,j] = (equil[1]/equil[0]) - opt_phiRb

tau_kappa_sweep_out = np.zeros((len(tau_range), len(kappa_range)))
for i, tau in enumerate(tqdm.tqdm(tau_range)):
    for j, kappa in enumerate(kappa_range):
        args = {'gamma_max':gamma_max, 
                'nu_max':  nu_max, 
                'tau': tau, 
                'Kd_TAA': Kd_TAA, 
                'Kd_TAA_star': Kd_TAA_star, 
                'kappa_max': kappa, 
                'phi_O': phi_O}
        equil = growth.model.equilibrate_ppGpp(args)
        tau_kappa_sweep_out[i,j] = (equil[1]/equil[0]) - opt_phiRb


# %%
fig, ax = plt.subplots(1, 2)
kd_out = ax[0].matshow(np.log10(np.abs(Kd_sweep_out)), cmap='mako', origin='lower', interpolation='none')
tau_kappa_out = ax[1].matshow(np.log10(np.abs(tau_kappa_sweep_out)), cmap='mako', origin='lower', interpolation='none')
# plt.colorbar()
# ax[0].plot(np.arange(0, resolution, 1), 'r--', lw=1)
ax[0].grid(False)
ax[1].grid(False)
# ax.set_xticks([0, 10, 20, 30, 40])
# ax.set_yticks([0, 10, 20, 30, 40])
# ax.set_xticklabels(['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$'])
# a.set_yticklabels(['$10^{-6}$', '$10^{-5}$', '$10^{-4}$', '$10^{-2}$', '$10^{0}$'])
ax[0].set_xlabel('charged-tRNA dissociation constant\n$K_D^{T_{AA}}$')
ax[0].set_ylabel('$K_D^{T_{AA}^*}$\nuncharged-tRNA\ndissociation constant')
ax[1].set_xlabel('charged-tRNA dissociation constant\n$K_D^{T_{AA}}$')
ax[1].set_ylabel(r'$\tau$' + '\ncharged-to-uncharged tRNA\nsensitivity parameter')
ax[1].set_xlabel('uncharged-tRNA transcription rate\n$\kappa_{max}$ [hr$^{-1}$]')

# fig.colorbar(_im, ax=ax, label='log\u2081\u2080 difference from optimal allocation')
# plt.savefig('../../figures/FPO_Kd_parameter_scan.pdf', bbox_inches='tight')
# %%
