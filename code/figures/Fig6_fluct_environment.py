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
nu_init = 0.05
nu_shift = 2 #1.83
total_time = 10 
shift_time = 2
phi_O = np.array([0.55])
phi_O_post = phi_O
dt = 0.0001
preshift_time = np.arange(0, shift_time + dt, dt)
postshift_time = np.arange(shift_time, total_time, dt)

# ppGpp params
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau'] 
phi_Rb = 0.2
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
                        kappa_max, phiO)
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                             init_params,np.arange(0, 200, dt), args=preshift_args)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, 
                          kappa_max, phi_O_post)
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
                        kappa_max, phiO)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA, Kd_TAA_star,
                          kappa_max, phi_O_post)

        # Integrate preshift
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, preshift_time, args=preshift_args)
        shift_params = preshift_out[-1]
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, postshift_time, args=postshift_args)
        postshift_out = postshift_out[1:]


        # Generate dataframes and concat
        preshift_df = pd.DataFrame(preshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        postshift_df = pd.DataFrame(postshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        preshift_df['time'] = preshift_time
        postshift_df['time'] = postshift_time[1:]
        dynamic_df = pd.concat([preshift_df, postshift_df], sort=False)
        dynamic_df['balance'] = dynamic_df['TAA_star'].values / dynamic_df['TAA'].values
        dynamic_df['phi_Rb'] = dynamic_df['balance'].values / (dynamic_df['balance'].values + tau) 
        dynamic_df['MRb_M'] = dynamic_df['MRb'].values / dynamic_df['M'].values
        dynamic_dfs.append(dynamic_df) 

        # Perform the shift for instant reallocation
        init_params = [M0, M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
        preshift_args = (gamma_max, nu_init, tau, Kd_TAA, Kd_TAA_star, 
                        kappa_max, phiO, False, False, False, True, init_phiRb)
        postshift_args = (gamma_max, nu_shift, tau, Kd_TAA, Kd_TAA_star,
                          kappa_max, phi_O, False, False, False, True, shift_phiRb)

        # Integrate preshift
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, preshift_time, args=preshift_args)
        shift_params = preshift_out[-1]
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, postshift_time, args=postshift_args)
        postshift_out = postshift_out[1:]

        # Generate dataframes and concat
        preshift_df = pd.DataFrame(preshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        postshift_df = pd.DataFrame(postshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        preshift_df['time'] = preshift_time
        preshift_df['phi_Rb'] = init_phiRb
        postshift_df['time'] = postshift_time[1:]
        postshift_df['phi_Rb'] = shift_phiRb
        instant_df = pd.concat([preshift_df, postshift_df], sort=False)
        instant_df['MRb_M'] = instant_df['MRb'].values / instant_df['M'].values
        instant_df['balance'] = instant_df['TAA_star'].values / instant_df['TAA'].values
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
# for g, d in instant_df.groupby(['set_phiO']): 
#         if g == 0:
#                 ls = '-'
#         else:
#                 ls = '-'
ax[0].plot(instant_df['time'], instant_df['phi_Rb'], '-', color=colors['primary_blue'],
                     label='instantaneous reallocation',  lw=1)
ax[1].plot(instant_df['time'], instant_df['MRb_M'], '-', color=colors['primary_blue'], lw=1)
inst_gr = np.log(instant_df['M'].values[1:]/instant_df['M'].values[:-1]) / dt
ax[2].plot(instant_df['time'].values[:-1], inst_gr, '-', color=colors['primary_blue'], lw=1)


# for g, d in dynamic_df.groupby(['set_phiO']):
#         if g == 0:
#                 ls = '--'
#         else:
#                 ls = '--'
ax[0].plot(dynamic_df['time'], dynamic_df['phi_Rb'], '--', color=colors['primary_red'],
                label='dynamic reallocation', lw=1)
ax[1].plot(dynamic_df['time'], dynamic_df['MRb_M'], '--', color=colors['primary_red'], lw=1)
inst_gr = np.log(dynamic_df['M'].values[1:]/dynamic_df['M'].values[:-1]) / dt
ax[2].plot(dynamic_df['time'].values[:-1], inst_gr, '--', color=colors['primary_red'], lw=1)


plt.tight_layout()
# plt.savefig('../../figures/Fig6_upshift_plots.pdf', bbox_inches='tight')


# %%
