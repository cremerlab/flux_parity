#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
import scipy.integrate
colors, palette = growth.viz.matplotlib_style()

# Load the dataset
data = pd.read_csv('../../data/upshift_mass_fraction.csv')


# Set the constants
gamma_max = 20 * 3600/ 7459 
nu_init = 0.5
nu_shift = 1.7 #1.83
total_time = 8
shift_time = 2

# ppGpp params
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3
phi_Rb = 0.5
phi_Rb_star = 0.00005
phi_Mb = 1 - phi_Rb - phi_Rb_star
OD_CONV = 1.5E17

# Init params
M0 = 0.001 * OD_CONV
M_Rb = phi_Rb * M0
M_Rb_star = phi_Rb_star * M0
M_Mb = phi_Mb * M0
T_AA = 0.0002
T_AA_star = 0.0002
k_Rb = 2  
kappa_max = (88 * 5 * 3600) / 1E9 #0.002
dt = 0.0001
time = np.arange(0, 10, dt)

init_params = [M_Rb, M_Rb_star, M_Mb, T_AA, T_AA_star]

preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, 
                k_Rb, False, True, True, True)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp_phi_O,
                             init_params, time, args=preshift_args)


rbstar_df = pd.DataFrame(preshift_out, columns=['M_Rb', 'M_Rb_star', 'M_Mb', 'T_AA', 'T_AA_star'])
rbstar_df['total_biomass'] = rbstar_df['M_Rb'].values + rbstar_df['M_Rb_star'].values + rbstar_df['M_Mb'].values
rbstar_df['relative_biomass'] = rbstar_df['total_biomass'] / M0
rbstar_df['mrb/m'] = rbstar_df['M_Rb'].values / rbstar_df['total_biomass'].values
rbstar_df['mrb_star/m'] = rbstar_df['M_Rb_star'].values / rbstar_df['total_biomass'].values
rbstar_df['tRNA_balance'] = rbstar_df['T_AA_star'].values / rbstar_df['T_AA']
rbstar_df['prescribed_phiRb'] = rbstar_df['tRNA_balance'] / (rbstar_df['tRNA_balance'] + tau)
rbstar_df['time'] = time


init_params = [M_Rb, M_Mb, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, 
                k_Rb, False, True, True, False)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, time, args=preshift_args)


preshift_df = pd.DataFrame(preshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
preshift_df['total_biomass'] = preshift_df['M_Rb'].values +  preshift_df['M_Mb'].values
preshift_df['relative_biomass'] = preshift_df['total_biomass'] / M0
preshift_df['mrb/m'] = preshift_df['M_Rb'].values / preshift_df['total_biomass'].values
preshift_df['tRNA_balance'] = preshift_df['T_AA_star'].values / preshift_df['T_AA']
preshift_df['prescribed_phiRb'] = preshift_df['tRNA_balance'] / (preshift_df['tRNA_balance'] + tau)
preshift_df['time'] = time



fig, ax = plt.subplots(2, 2, figsize=(6,6))
ax[0,0].plot(preshift_df['time'], preshift_df['relative_biomass'], 'k-')
ax[1,1].plot(preshift_df['time'], preshift_df['mrb/m'], 'k-')
ax[0,1].plot(preshift_df['time'], preshift_df['prescribed_phiRb'], 'k-')
ax[0, 0].set(yscale='log')
ax[0,0].plot(rbstar_df['time'], rbstar_df['relative_biomass'], 'r--')
ax[0,1].plot(rbstar_df['time'], rbstar_df['prescribed_phiRb'], 'r--')
ax[1,0].plot(rbstar_df['time'], rbstar_df['mrb_star/m'], 'r--')
ax[1,1].plot(rbstar_df['time'], rbstar_df['mrb/m'], 'r--')

#%%
preshift_star_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, True)
postshift_star_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, True)
init_star_params = [M_Rb, M_Mb, 0, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, False)
postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, False)
init_params = [M_Rb, M_Mb, T_AA, T_AA_star]

preshift_star_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_star_params, np.arange(0, 200, dt), args=preshift_star_args)
postshift_star_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_star_params, np.arange(0, 200, dt), args=postshift_star_args)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=preshift_args)
postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=postshift_args)

# Star processing 
preshift_star_out = preshift_star_out[-1]
postshift_star_out = postshift_star_out[-1]
preshift_out = preshift_out[-1]
postshift_out = postshift_out[-1]

init_star_phiRb = (preshift_star_out[0]) / (preshift_star_out[0] + preshift_star_out[1] + preshift_star_out[2])
init_star_phiMb = (preshift_star_out[2]) / (preshift_star_out[0] + preshift_star_out[1] + preshift_star_out[2])
init_star_phiRb_star = (preshift_star_out[1]) / (preshift_star_out[0] + preshift_star_out[1] + preshift_star_out[2])
shift_star_phiRb = (postshift_star_out[0] ) / (postshift_star_out[0] + postshift_star_out[1] + postshift_star_out[2])
shift_star_phiMb = (postshift_star_out[2]) / (postshift_star_out[0] + postshift_star_out[1] + postshift_star_out[2])
shift_star_phiRb_star = (postshift_star_out[1]) / (postshift_star_out[0] + postshift_star_out[1] + postshift_star_out[2])
init_star_T_AA = preshift_star_out[-2]
init_star_T_AA_star = preshift_star_out[-1]

init_phiRb = (preshift_out[0]) / (preshift_out[0] + preshift_out[1])
init_phiMb = (preshift_out[2]) / (preshift_out[0] + preshift_out[1])
shift_phiRb = (postshift_out[0]) / (postshift_out[0] + postshift_out[1])
shift_phiMb = (postshift_out[2]) / (postshift_out[0] + postshift_out[1])
shift_phiRb_star = (postshift_out[1]) / (postshift_out[0] + postshift_out[1])
init_T_AA = preshift_out[-2]
init_T_AA_star = preshift_out[-1]

# Do the shift
init_star_params = [M0 * init_star_phiRb, M0 * init_star_phiRb_star, M0 * init_star_phiMb, init_star_T_AA, init_star_T_AA_star]
init_star_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, (init_star_phiRb + init_star_phiRb_star), k_Rb, False, True, True, True)
inst_preshift_star_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, (init_star_phiRb + init_star_phiRb_star), k_Rb, False, False, True, True)
inst_postshift_star_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, kappa_max, (shift_star_phiRb + shift_star_phiRb_star), k_Rb, False, False, True, True)
dynamic_star = growth.model.nutrient_shift_ppGpp(nu_init, nu_shift, shift_time, 
                                            init_star_params, init_star_args,
                                            total_time, maturation=True)
instant_star = growth.model.nutrient_shift_ppGpp(nu_init, nu_shift, shift_time, 
                                            init_star_params, inst_preshift_star_args,
                                            total_time, maturation=True, postshift_args=inst_postshift_star_args)

# init_params = [M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
# init_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, False)
# shift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, kappa_max, 0, k_Rb, False, True, True, False)
# instant = growth.model.nutrient_shift_ppGpp(nu_init, nu_shift, shift_time, 
#                                             init_params, init_args,
#                                             total_time, postshift_args=shift_args)


fig, ax = plt.subplots(1, 1)
ax.set_xlabel('time from up-shift [min]')
ax.set_ylabel('ribosome mass fraction $M_{Rb}/M$')
ax.plot((dynamic_star['time'].values - shift_time)*60, dynamic_star['realized_phiR'].values,  
        '-', lw=1, color=colors['primary_black'], label='dynamic reallocation (with maturation)')
ax.plot((instant_star['time'].values - shift_time) * 60, instant_star['realized_phiR'].values,  
        '-', lw=1, color=colors['primary_blue'], label='instantaneous reallocation (with maturation)')


# Plot the Bremer data
_data = data[data['source']=='Erickson et al., 2017']
ax.plot(_data['time_from_shift_min'], _data['mass_fraction'], 'o', ms=4, label=_data['source'].values[0])
# ax.set_xlim([0, 100])
ax.legend()
# %%
# Compute the instantaneous growth rate
dynamic_gr = np.log(dynamic_star['total_biomass'].values[1:] / dynamic_star['total_biomass'].values[:-1]) / np.diff(dynamic_star['time'].values)[1]
instant_gr = np.log(instant_star['total_biomass'].values[1:] / instant_star['total_biomass'].values[:-1]) / np.diff(instant_star['time'].values)[1]

# %%
lam_data = pd.read_csv('../../data/Erickson2017_Fig1B.csv')
plt.plot((dynamic_star['time'].values[1:] - shift_time), dynamic_gr, 'k-', lw=1)
plt.plot((instant_star['time'].values[1:] - shift_time), instant_gr, '-', lw=1, color=colors['primary_blue'])
plt.plot(lam_data['time_from_shift_hr'], lam_data['growth_rate_hr'], 'o', ms=4)

# %% Exploring phiO shift

# Set the constants
gamma_max = 20 * 3600/ 7459 
nu_init = 5 
nu_shift = 1.7 #1.83
total_time = 20 
shift_time = 2

# ppGpp params
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3
phi_O = 0.25
phi_R = 0.1
phi_Mb = 1 - phi_Rb - phi_O
OD_CONV = 1.5E17

# Init params
M0 = 0.001 * OD_CONV
M_Rb = phi_Rb * M0
M_Mb = phi_Mb * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002
dt = 0.0001
time = np.arange(0, 10, dt)

init_params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, phi_O, kappa_max, 0, 
                 False, True, True)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp_phi_O,
                             init_params, time, args=preshift_args)


phiO_df = pd.DataFrame(preshift_out, columns=['M_Rb', 'M_Rb_star', 'M_Mb', 'T_AA', 'T_AA_star'])
phiO_df['total_biomass'] = phiO_df['M_Rb'].values + phiO_df['M_Rb_star'].values + phiO_df['M_Mb'].values
phiO_df['relative_biomass'] = phiO_df['total_biomass'] / M0
phiO_df['mrb/m'] = phiO_df['M_Rb'].values / phiO_df['total_biomass'].values
phiO_df['mrb_star/m'] = phiO_df['M_Rb_star'].values / phiO_df['total_biomass'].values
phiO_df['tRNA_balance'] = phiO_df['T_AA_star'].values / phiO_df['T_AA']
phiO_df['prescribed_phiRb'] = phiO_df['tRNA_balance'] / (phiO_df['tRNA_balance'] + tau)
phiO_df['time'] = time

# %%
fig, ax = plt.subplots(2, 2, figsize=(6,6))
ax[0,0].plot(preshift_df['time'], preshift_df['relative_biomass'], 'k-')
ax[1,1].plot(preshift_df['time'], preshift_df['mrb/m'], 'k-')
ax[0,1].plot(preshift_df['time'], preshift_df['prescribed_phiRb'], 'k-')
ax[0, 0].set(yscale='log')
# ax[0,0].plot(rbstar_df['time'], rbstar_df['relative_biomass'], 'r--')
# ax[0,1].plot(rbstar_df['time'], rbstar_df['prescribed_phiRb'], 'r--')
# ax[1,0].plot(rbstar_df['time'], rbstar_df['mrb_star/m'], 'r--')
# ax[1,1].plot(rbstar_df['time'], rbstar_df['mrb/m'], 'r--')

# %%
nu_max_1 = 0.65
nu_max_2 = 1.75 
nu_max = [nu_max_1, nu_max_2]
prefactors = [[1, 0], [0, 1]]
# ppGpp params
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3

phi_R = 0.1
phi_Mb_2 = 0.25
phi_Mb_1 = 1 - phi_Rb - phi_Mb_2
OD_CONV = 1.5E17

# Init params
M0 = 0.001 * OD_CONV
M_Rb = phi_Rb * M0
M_Mb_1 = phi_Mb_1 * M0
M_Mb_2 = phi_Mb_2 * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002
dt = 0.0001
time = np.arange(0, 10, dt)


preshift_args = (gamma_max, nu_max, prefactors[0], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, 0, False, True, True)
postshift_args = (gamma_max, nu_max, prefactors[1], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, 0, False, True, True)
init_params_1 = [M0, M_Rb, M_Mb_1, M_Mb_2, T_AA, T_AA_star]
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp_shift,
                             init_params, np.arange(0, 200, dt), args=preshift_args)
postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp_shift,
                             init_params, np.arange(0, 200, dt), args=postshift_args)
preshift_out = preshift_out[-1]
postshift_out = postshift_out[-1]
init_phiRb = preshift_out[1] / preshift_out[0]
init_phiMb_1 = preshift_out[2] / preshift_out[0]
init_phiMb_2 = preshift_out[3] / preshift_out[0]
init_phiO = 1 - init_phiRb - init_phiMb
shift_phiRb = postshift_out[1] / postshift_out[0]
shift_phiMb_1 = postshift_out[2] / postshift_out[0]
shift_phiMb_2 = postshift_out[3] / postshift_out[0]
init_T_AA = preshift_out[-2]
init_T_AA_star = preshift_out[-1]



init_params = [M0, M0 * init_phiRb, M0 * init_phiMb_1, M0 * init_phiMb_2, init_T_AA, init_T_AA_star]
init_args = (gamma_max, nu_max, prefactors[0], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, 0, False, True, True)
shift_args = (gamma_max, nu_max, prefactors[1], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, 0, False, True, True)
shift = growth.model.nutrient_shift_ppGpp(nu_max_1, nu_max_2, shift_time, 
                                                init_params, init_args,
                                                total_time, postshift_args=shift_args)

init_args = (gamma_max, nu_max, prefactors[0], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, init_phiRb, False, False, True)
shift_args = (gamma_max, nu_max, prefactors[1], tau, Kd_TAA_star, Kd_TAA, phi_Mb_2, kappa_max, shift_phiRb, False, False, True)
inst = growth.model.nutrient_shift_ppGpp(nu_max_1, nu_max_2, shift_time, 
                                                init_params, init_args,
                                                total_time, postshift_args=shift_args)



fig, ax = plt.subplots(3, 3, figsize=(6,7))
for a in ax.ravel():
        a.set_xlabel('time from shift [hr]')
ax[0, 0].set_ylabel('$\phi_{Rb}$')
ax[0, 1].set_ylabel('$\phi_{Mb,1}$')
ax[0, 2].set_ylabel('$\phi_{Mb,2}$')
ax[1, 0].set_ylabel('$M_{Rb}/M$')
ax[1, 1].set_ylabel('$M_{Mb,1}/M$')
ax[1, 2].set_ylabel('$M_{Mb,2}/M$')
ax[0,0].plot(shift['time']- shift_time, shift['prescribed_phiR'])
ax[0,1].plot(shift['time'] - shift_time, shift['prescribed_phiMb1'])
ax[0,2].plot(shift['time'] - shift_time, shift['prescribed_phiMb2'])
ax[1,0].plot(shift['time'] - shift_time, shift['realized_phiR'])
ax[1,1].plot(shift['time'] - shift_time, shift['realized_phiMb1'])
ax[1,2].plot(shift['time'] - shift_time, shift['realized_phiMb2'])
ax[1, 0].plot(_data['time_from_shift_min'] / 60, _data['mass_fraction'], 'o', ms=3)
ax[0,0].plot(inst['time']- shift_time, inst['prescribed_phiR'])
ax[0,1].plot(inst['time'] - shift_time, inst['prescribed_phiMb1'])
ax[0,2].plot(inst['time'] - shift_time, inst['prescribed_phiMb2'])
ax[1,0].plot(inst['time'] - shift_time, inst['realized_phiR'])
ax[1,1].plot(inst['time'] - shift_time, inst['realized_phiMb1'])
ax[1,2].plot(inst['time'] - shift_time, inst['realized_phiMb2'])
_data = data[data['source']=='Erickson et al., 2017']
ax[1, 0].plot(_data['time_from_shift_min'] / 60, _data['mass_fraction'], 'o', ms=3)

inst_gr = np.log(inst['total_biomass'].values[1:]/inst['total_biomass'].values[:-1]) / (np.diff(inst['time'].values)[1])
shift_gr = np.log(shift['total_biomass'].values[1:]/shift['total_biomass'].values[:-1]) / (np.diff(shift['time'].values)[1])

ax[2, 0].plot(inst['time'].values[1:] - shift_time, inst_gr)
ax[2, 0].plot(shift['time'].values[1:] - shift_time, shift_gr)
ax[2, 0].plot(lam_data['time_from_shift_hr'], lam_data['growth_rate_hr'], 'o', ms=4)

# %%

# %%
