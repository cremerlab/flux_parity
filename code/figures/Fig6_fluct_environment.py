#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
import tqdm
import scipy.integrate
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Set the constants
gamma_max = const['gamma_max']
nu_init = 0.5
nu_shift = 2 #1.83
total_time = 8
shift_time = 2
phi_O = np.array([0.35])
phi_O_post = phi_O - 0.1

# ppGpp params
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau'] 
phi_Rb = 0.5
phi_Mb = 1 - phi_Rb - phi_O
# Init params
M0 = 1E9
M_Rb = phi_Rb * M0
M_Mb = phi_Mb * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = const['kappa_max']
dt = 0.0001
time = np.arange(0, 10, dt)

dynamic_dfs = []
instant_dfs = []
for i, phiO in enumerate(tqdm.tqdm(phi_O)):
        M_Mb = (1 - phi_Rb - phiO) * M0
        init_params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
        preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA,
                        kappa_max, phiO, 0, False, True, True)
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                             init_params,np.arange(0, 200, dt), args=preshift_args)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, 
                          kappa_max, phi_O_post[i], 0, False, True, True)
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=postshift_args)


        # Compute the initial states
        preshift_out = preshift_out[-1]
        postshift_out = postshift_out[-1]
        init_phiRb = preshift_out[1] / preshift_out[0]
        init_phiMb = preshift_out[2] / preshift_out[0]
        init_phiO = 1 - init_phiRb - init_phiMb
        shift_phiRb = postshift_out[1] / postshift_out[0]
        shift_phiMb = postshift_out[2] / postshift_out[0]
        shift_phiO = 1 - shift_phiRb - shift_phiMb 
        init_T_AA = preshift_out[-2]
        init_T_AA_star = preshift_out[-1]

        # Perform the shift for dynamic reallocation
        init_params = [M0, M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
        preshift_args = (gamma_max, nu_init, tau, Kd_TAA, Kd_TAA_star, 
                        kappa_max, phiO, 0, False, True, True)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA, Kd_TAA_star,
                          kappa_max, phi_O_post[i], 0, False, True, True)
        dynamic_df = growth.model.nutrient_shift_ppGpp(nu_init, nu_shift, shift_time,
                                                init_params, preshift_args,
                                                total_time, postshift_args)
        dynamic_df['phi_O'] = phiO
        dynamic_df.loc[dynamic_df['time'] >= shift_time, 'phi_O'] = 0
        dynamic_df['set_phiO'] = phiO
        dynamic_dfs.append(dynamic_df)

        # Perform the shift for instant reallocation
        preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA,
                        kappa_max, phiO, init_phiRb, False, False, True)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, 
                          kappa_max, phi_O_post[i], shift_phiRb, False, False, True)
        instant_df = growth.model.nutrient_shift_ppGpp(nu_init, nu_shift, shift_time,
                                                init_params, preshift_args,
                                                total_time, postshift_args)
        instant_df['phi_O'] = phiO
        instant_df.loc[instant_df['time'] >= shift_time, 'phi_O'] = 0
        instant_df['set_phiO'] = phiO

        instant_dfs.append(instant_df)


dynamic_df = pd.concat(dynamic_dfs, sort=False)
dynamic_df['time'] -= shift_time
instant_df = pd.concat(instant_dfs, sort=False)
instant_df['time'] -= shift_time
#%%
fig, ax = plt.subplots(1, 3, figsize=(6,2), sharex=True)

# Add labels
ax[0].set_xlabel('time from upshift [hr]')
ax[1].set_xlabel('time from upshift [hr]')
ax[2].set_xlabel('time from upshift [hr]')
ax[0].set_ylabel('$\phi_{Rb}$')
ax[1].set_ylabel('$M_{Rb}/M$')
ax[2].set_ylabel('$\lambda$ [hr$^{-1}$]')
ax[0].set_ylim([0, 1])
ax[1].set_ylim([0, 0.25])
ax[2].set_ylim([0, 1.25])
for g, d in instant_df.groupby(['set_phiO']): 
        if g == 0:
                ls = '-'
        else:
                ls = '-'
        ax[0].plot(d['time'], d['prescribed_phiR'], ls, color=colors['primary_blue'],
                     label='instantaneous reallocation',  lw=1)
        ax[1].plot(d['time'], d['realized_phiR'], ls, color=colors['primary_blue'], lw=1)
        inst_gr = np.log(d['total_biomass'].values[1:]/d['total_biomass'].values[:-1]) / (d['time'].values[1:] - d['time'].values[:-1])
        ax[2].plot(d['time'].values[:-1], inst_gr, ls, color=colors['primary_blue'], lw=1)


for g, d in dynamic_df.groupby(['set_phiO']):
        if g == 0:
                ls = '--'
        else:
                ls = '--'
        ax[0].plot(d['time'], d['prescribed_phiR'], ls, color=colors['primary_red'],
                        label='dynamic reallocation', lw=1)
        ax[1].plot(d['time'], d['realized_phiR'], ls, color=colors['primary_red'], lw=1)
        inst_gr = np.log(d['total_biomass'].values[1:]/d['total_biomass'].values[:-1]) / (d['time'].values[1:] - d['time'].values[:-1])
        ax[2].plot(d['time'].values[:-1], inst_gr, ls, color=colors['primary_red'], lw=1)


plt.tight_layout()
plt.savefig('../../figures/Fig6_upshift_plots.pdf', bbox_inches='tight')


# %%
