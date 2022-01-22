#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.viz
import growth.integrate
import pickle 
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
#%%
# Define constant params
gamma_max = const['gamma_max']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
Kd_cpc = const['Kd_cpc']
nu_max = 4

# Compute the optimium
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)

# Define parameter ranges
resolution = 75 
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
        equil = growth.integrate.equilibrate_FPM(args, tol=2)
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
        equil = growth.integrate.equilibrate_FPM(args, tol=2)
        tau_kappa_sweep_out[i,j] = (equil[1]/equil[0]) - opt_phiRb

#%%
np.save('../../figures/Kd_parameter_sweep.pkl', Kd_sweep_out)
np.save('../../figures/tau_kappa_parameter_sweep.pkl', tau_kappa_sweep_out)

#%%
Kd_sweep_out = np.load('../../figures/Kd_parameter_sweep.pkl')
tau_kappa_sweep_out = np.load('../../figures/tau_kappa_parameter_sweep.pkl')
# %%
fig, ax = plt.subplots(1, 2)
Kd_sweep_scaled = np.log10(np.abs(Kd_sweep_out))
tau_kappa_sweep_scaled = np.log10(np.abs(tau_kappa_sweep_out))
kd_out = ax[0].imshow(Kd_sweep_scaled,
                       cmap='mako', 
                       origin='lower', 
                       interpolation='none',
                       vmin=-3,
                       vmax=-1)
tau_kappa_out = ax[1].imshow(tau_kappa_sweep_scaled,
                            cmap='mako', 
                            origin='lower', 
                            interpolation='none', 
                            vmin=-3,
                            vmax=-1)
# plt.colorbar()
ax[0].grid(False)
ax[1].grid(False)

# Set up the indices and labels for the Kd_sweep
Kd_inds = [np.where(np.round(np.log10(Kd_TAA_range), decimals=1) == i)[0][0] for i in range(-5,-1)] 
labels = ["10$^{%s}$" % i for i in range(-5, -1)]
ax[0].set_xticks(Kd_inds)
ax[0].set_xticklabels(labels)
ax[0].set_yticks(Kd_inds)
ax[0].set_yticklabels(labels)

# Set up indices and labels for tau kappa sweep
tau_inds = [np.where(np.round(np.log10(tau_range), decimals=1)==i)[0][0] for i in range(-3, 2)]
tau_labels = ["10$^{%s}$" % i for i in range(-3, 2)]
kappa_inds = [np.where(np.round(np.log10(kappa_range), decimals=1)==i)[0][0] for i in range(-5, 1)]
kappa_labels = ["10$^{%s}$" % i for i in range(-5, 1)]
ax[1].set_xticks(kappa_inds)
ax[1].set_yticks(tau_inds)
ax[1].set_xticklabels(kappa_labels)
ax[1].set_yticklabels(tau_labels)

ax[0].set_xlabel('tRNA uncharging Michaelis-Menten constant\n$K_D^{tRNA^*}}$ [abundance units]')
ax[0].set_ylabel('$K_M^{tRNA}$ [abundance units] \ntRNA charging Michaelis-Menten constant')

ax[1].set_ylabel(r'$\tau$' + '\ncharged-to-uncharged tRNA\nsensitivity parameter')
ax[1].set_xlabel('uncharged-tRNA transcription rate\n$\kappa_{max}$ [abundance units $\cdot$ hr$^{-1}$]')

plt.tight_layout()
cbar = fig.colorbar(kd_out, ax=ax[:], label='absolute difference from optimal allocation',
            location='bottom', shrink=0.8, ticks=[-3, -2, -1], )
cbar.ax.set_xticklabels(['$10^{-3}$', '$10^{-2}$', '$10^{-1}$'])  # vertically oriented colorbar
fig.text(0, 0.91, '(A)', fontsize=8, fontweight='bold')
fig.text(0.5, 0.91, '(B)', fontsize=8, fontweight='bold')
plt.savefig('../../figures/FigS7_FPM_parameter_sweep.pdf', bbox_inches='tight')
# %%
