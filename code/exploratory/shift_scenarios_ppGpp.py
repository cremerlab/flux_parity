#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import scipy.integrate
colors, palette = growth.viz.matplotlib_style()

# %%
nu_init =   1 
nu_shift = 5 

# Set the constants for all scenarios
gamma_max = 20 * 3600 / 7459
OD_CONV = 1.5E17
shift_time = 1

# Set for ppGpp scenario
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002

# Set for optimal allocation scenario
Kd = 0.012

# set the initial conditions for the integration
M0 = 0.001 * OD_CONV
M_Rb = 0.5 * M0
M_Mb = 0.5 * M0

# Set the two time ranges 
dt = 0.0001
preshift = np.arange(0, shift_time,dt)
postshift = np.arange(shift_time - dt, 7, dt)

# Set the optimal and constant phiRbs
init_params = [M_Rb, M_Mb, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=preshift_args)
postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=postshift_args)

preshift_out = preshift_out[-1]
postshift_out = postshift_out[-1]
init_phiRb = (preshift_out[0]) / (preshift_out[0] + preshift_out[1])
shift_phiRb = (postshift_out[0]) / (postshift_out[0] + postshift_out[1])
init_phiMb = 1 - init_phiRb
shift_phiMb = 1 - shift_phiRb
init_T_AA = preshift_out[2]
init_T_AA_star = preshift_out[3]

init_params = [M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
# Compute the constant scenario
const_preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, False, True, kappa_max, init_phiRb)
const_postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, False, True, kappa_max, init_phiRb)
const_preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp, init_params,
                                            preshift, args=const_preshift_args)
preshift_df = pd.DataFrame(const_preshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
const_postshift_params = const_preshift_out[-1]
const_postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp, const_postshift_params,
                                            postshift, args=const_postshift_args)
postshift_df = pd.DataFrame(const_postshift_out[1:], columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] =  postshift[1:]

const_shift_df = pd.concat([preshift_df, postshift_df])
const_shift_df['total_biomass'] = const_shift_df['M_Rb'].values + const_shift_df['M_Mb'].values
const_shift_df['relative_biomass'] = const_shift_df['total_biomass'].values / M0
const_shift_df['prescribed_phiR'] = init_phiRb
const_shift_df['realized_phiR'] = const_shift_df['M_Rb'].values / const_shift_df['total_biomass'].values
const_shift_df['gamma'] = gamma_max * const_shift_df['T_AA_star'].values / (const_shift_df['T_AA_star'] + Kd_TAA_star)

const_inst_gr = np.log(const_shift_df['total_biomass'].values[1:]/const_shift_df['total_biomass'].values[:-1])/dt

# Optimal scenario


opt_preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, False, True, kappa_max, init_phiRb)
opt_postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, False, True, kappa_max, shift_phiRb)
opt_preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp, init_params,
                                            preshift, args=opt_preshift_args)
preshift_df = pd.DataFrame(const_preshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
preshift_df['prescribed_phiR'] = init_phiRb
opt_postshift_params = opt_preshift_out[-1]
opt_postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp, opt_postshift_params,
                                            postshift, args=opt_postshift_args)
postshift_df = pd.DataFrame(opt_postshift_out[1:], columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] =  postshift[1:]
postshift_df['prescribed_phiR'] = shift_phiRb

opt_shift_df = pd.concat([preshift_df, postshift_df])
opt_shift_df['total_biomass'] = opt_shift_df['M_Rb'].values + opt_shift_df['M_Mb'].values
opt_shift_df['relative_biomass'] = opt_shift_df['total_biomass'].values / M0
opt_shift_df['realized_phiR'] = opt_shift_df['M_Rb'].values / opt_shift_df['total_biomass'].values
opt_shift_df['gamma'] = gamma_max * opt_shift_df['T_AA_star'].values / (opt_shift_df['T_AA_star'] + Kd_TAA_star)

opt_inst_gr = np.log(opt_shift_df['total_biomass'].values[1:]/opt_shift_df['total_biomass'].values[:-1])/dt

# Figure out where to start teh ppGpp model
# init_params = [M_Rb, M_Mb, T_AA, T_AA_star]
# preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, kappa_max)
# postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, True, kappa_max)
# out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
#                              init_params, np.arange(0, 150, dt), args=preshift_args)
# out = out[-1]
# ppGpp_init_phiRb = (out[0]) / (out[0] + out[1])
# print(ppGpp_init_phiRb)
# ppGpp_init_phiMb = 1 - ppGpp_init_phiRb
# init_T_AA = out[2]
# init_T_AA_star = out[3]
# Compute the preshift ppGpp
ppGpp_preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
ppGpp_postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)

ppGpp_preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                         init_params, preshift, args=ppGpp_preshift_args)
preshift_df = pd.DataFrame(ppGpp_preshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
ppGpp_shift_params = ppGpp_preshift_out[-1]
ppGpp_postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                        ppGpp_shift_params, postshift, args=ppGpp_postshift_args)
postshift_df = pd.DataFrame(ppGpp_postshift_out[1:], columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] = postshift[1:]
ppGpp_shift_df = pd.concat([preshift_df, postshift_df])

# Compute properties
ppGpp_shift_df['total_biomass'] = ppGpp_shift_df['M_Rb'].values + ppGpp_shift_df['M_Mb'].values
ppGpp_shift_df['relative_biomass'] = ppGpp_shift_df['total_biomass'].values / M0
ppGpp_shift_df['tRNA_balance'] = ppGpp_shift_df['T_AA_star'].values / ppGpp_shift_df['T_AA'].values
ppGpp_shift_df['prescribed_phiR'] = ppGpp_shift_df['tRNA_balance'].values / (ppGpp_shift_df['tRNA_balance'].values + tau)
ppGpp_shift_df['realized_phiR'] = ppGpp_shift_df['M_Rb'].values / ppGpp_shift_df['total_biomass'].values
ppGpp_shift_df['gamma'] = gamma_max * ppGpp_shift_df['T_AA_star'].values / (ppGpp_shift_df['T_AA_star'].values + Kd_TAA_star)

ppGpp_inst_gr = np.log(ppGpp_shift_df['total_biomass'].values[1:] / ppGpp_shift_df['total_biomass'].values[:-1])/dt



# palette = sns.color_palette('crest', n_colors=len(nu_max) + 10)

# %%

fig, ax = plt.subplots(4, 1, figsize=(6, 6), sharex=True)
ax[0].set( yscale='log',)
            # ylim=[1, 100])
ax[1].set(ylabel='ribosomal allocation $\phi_{Rb}$',
          ylim=[0, 1])
ax[2].set(ylabel='ribosome content $M_{Rb}/M$',
         ylim=[0, 1])
ax[3].set(ylabel='translation rate [AA / s]',
          xlabel='time [hr]',
          ylim=[0, 20])


# ppGpp model
ax[0].plot(ppGpp_shift_df['time_hr'].values[:-1], ppGpp_inst_gr, '-', lw=1, color=colors['primary_red'])
# ax[0].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['relative_biomass'], '-', 
            # lw=1, label='dynamic re-allocation', color=colors['primary_red'])
ax[1].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['prescribed_phiR'], '-', lw=1, color=colors['primary_red'])
ax[2].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['realized_phiR'], '-', lw=1, color=colors['primary_red'])
ax[3].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['gamma'] * 7459 / 3600, '-',  lw=1, color=colors['primary_red'])

# Constant model
# ax[0].plot(const_shift_df['time_hr'], const_shift_df['relative_biomass'], 'k-', 
            # lw=1, label='fixed allocation')

ax[0].plot(const_shift_df['time_hr'].values[:-1], const_inst_gr, 'k-', lw=1)
ax[1].plot(const_shift_df['time_hr'], const_shift_df['prescribed_phiR'], 'k-', lw=1)
ax[2].plot(const_shift_df['time_hr'], const_shift_df['realized_phiR'], 'k-', lw=1)
ax[3].plot(const_shift_df['time_hr'], const_shift_df['gamma'] * 7459 / 3600, 'k-',  lw=1)

# Optimal model
# ax[0].plot(opt_shift_df['time_hr'], opt_shift_df['relative_biomass'], '-', 
            # lw=1, label='instantaneous  re-allocation', color=colors['primary_blue'])

ax[0].plot(opt_shift_df['time_hr'].values[:-1], opt_inst_gr, '-', lw=1, color=colors['primary_blue'])
ax[1].plot(opt_shift_df['time_hr'], opt_shift_df['prescribed_phiR'], '-', color=colors['primary_blue'],lw=1)
ax[2].plot(opt_shift_df['time_hr'], opt_shift_df['realized_phiR'], '-', color=colors['primary_blue'], lw=1)
ax[3].plot(opt_shift_df['time_hr'], opt_shift_df['gamma'] * 7459 / 3600, '-', color=colors['primary_blue'], lw=1)


plt.tight_layout()

# ax.set(yscale='log')
# Add a line indicating the time of the shift
for a in ax:
    a.vlines(shift_time, a.get_ylim()[0], a.get_ylim()[1], lw=5, color=colors['pale_black'], 
            alpha=0.75, label='__nolegend__')
ax[0].legend()
# plt.savefig('../figures/ppGpp_shift_strategies.pdf', bbox_inches='tight')
# %%

# %%
fig, ax = plt.subplots(1, 1)

ax.plot(opt_shift_df['time_hr'], opt_shift_df['T_AA_star'], 'b-', lw=1)
ax.plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['T_AA_star'], 'r-', lw=1)
ax.plot(const_shift_df['time_hr'], const_shift_df['T_AA_star'], 'k-', lw=1)
# %%
