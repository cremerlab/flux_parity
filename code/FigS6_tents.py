#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.integrate
import growth.model 
import growth.viz 
import tqdm
colors, palette = growth.viz.matplotlib_style();
const = growth.model.load_constants()
# %%
# Define the constants 
gamma_max = const['gamma_max']
nu_max = 10 * gamma_max 
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']
# %%
# Set the range of phiR
phi_Rb = np.arange(0.001, 1 - phi_O - 0.001, 0.01)
dt = 0.001
df = pd.DataFrame({})
for i, phi in enumerate(tqdm.tqdm(phi_Rb)):
    # Equilibrate the model at this phiR
    args = {'gamma_max': gamma_max,
            'nu_max': nu_max,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O,
            'tau': tau,
            'dynamic_phiRb': False,
            'phiRb':  phi}
    _out = growth.integrate.equilibrate_FPM(args, t_return=2)
    out = _out[-1]
    lam = np.log(_out[-1][0] / _out[0][0]) / dt

    # Compute the various properties
    gamma = gamma_max * (out[-1] / (out[-1] + Kd_TAA_star))
    nu = nu_max * (out[-2] / (out[-2] + Kd_TAA))
    ratio = out[-1] / out[-2]

    results = {'gamma': gamma,
               'nu': nu, 
               'TAA': out[-2],
               'TAA_star': out[-1],
               'lam': gamma * phi, 
               'tot_tRNA': out[-1] + out[-2],
               'kappa': kappa_max * phi,
               'phi_Rb': phi,
               'metabolic_flux': nu * (1 - phi_O - phi),
               'translational_flux': gamma * phi}
    df = df.append(results, ignore_index=True)

#%% 
# find the optimal values
args = {'gamma_max': gamma_max,
            'nu_max': nu_max,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O,
            'tau': tau}
out = growth.integrate.equilibrate_FPM(args)
opt_phiRb = out[1]/out[0]
opt_TAA = out[-2]
opt_TAA_star = out[-1]
tot_tRNA = out[-2] + out[-1]
opt_gamma = gamma_max * opt_TAA_star / (opt_TAA_star + Kd_TAA_star)
opt_nu = nu_max * opt_TAA / (opt_TAA + Kd_TAA)
opt_kappa = kappa_max * (out[-1]/out[-2]) / ((out[-1]/out[-2]) + tau)
opt_lam =  opt_gamma * opt_phiRb
opt_translational_flux = opt_gamma * phi_Rb * (1 - tot_tRNA)
opt_metabolic_flux = opt_nu * (1 - phi_O - phi_Rb) + opt_kappa

#%% Compute the other scenarios
low_TAA_star = 0.1 * opt_TAA_star
low_TAA = 10 * opt_TAA
low_gamma = gamma_max * low_TAA_star / (low_TAA_star + Kd_TAA_star)
low_nu = nu_max * low_TAA / (low_TAA + Kd_TAA)
low_metabolic_flux = low_nu * (1 - phi_O - phi_Rb) + opt_kappa 
low_translational_flux = low_gamma * phi_Rb * (1 - tot_tRNA)

high_TAA_star = 10 * opt_TAA_star
high_TAA = 0.1 * opt_TAA
high_gamma = gamma_max * high_TAA_star / (high_TAA_star + Kd_TAA_star)
high_nu = nu_max * high_TAA / (high_TAA + Kd_TAA)
high_metabolic_flux = high_nu * (1 - phi_O - phi_Rb) + opt_kappa 
high_translational_flux = high_gamma * phi_Rb * (1 - tot_tRNA)

# %%
fig, ax = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(6.5, 2))
# Add labels
ax[0].set_ylabel('rate [hr$^{-1}$]')
for i in range(3):
    ax[i].plot(df['phi_Rb'], df['gamma'] * df['phi_Rb'], '-', lw=2, color=colors['primary_black'],
                label='steady-state growth')
    ax[i].set_xlabel('allocation towards ribosomes\n$\phi_{Rb}$')
ax[1].plot(phi_Rb, opt_translational_flux, '--', color=colors['primary_gold'], lw=2,
            label='translational flux')
ax[1].plot(phi_Rb, opt_metabolic_flux, '--', color=colors['primary_purple'], lw=2,
            label='metabolic flux rate')
ax[0].plot(phi_Rb, low_translational_flux, '--', color=colors['primary_gold'], lw=2,
            label='translational flux')
ax[0].plot(phi_Rb, low_metabolic_flux, '--', color=colors['primary_purple'], lw=2,
            label='metabolic flux')
ax[2].plot(phi_Rb, high_translational_flux, '--', color=colors['primary_gold'], lw=2,
            label='translational flux')
ax[2].plot(phi_Rb, high_metabolic_flux, '--', color=colors['primary_purple'], lw=2,
            label='metabolic flux')


ax.plot(df['phi_Rb'], df['translational_flux'])
ax.plot(df['phi_Rb'], df['gamma'])
ax.plot(opt_phiRb, opt_lam, 'o')
ax[1].set_ylim([0, 1.5])
plt.tight_layout()
plt.savefig('../figures/supplement_text/plots/FigS6_FPM_tents.pdf')