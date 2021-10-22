#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import scipy.integrate
colors, palette = growth.viz.matplotlib_style()

# %%
nu_init = 10  
nu_shift = 4 

# Set the constants for all scenarios
gamma_max = 9.65
OD_CONV = 1.5E17
shift_time = 1

# Set for ppGpp scenario
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 4
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002

# Set for optimal allocation scenario
Kd = 0.025

# Set the optimal and constant phiRbs
init_phiRb = growth.model.phi_R_optimal_allocation(gamma_max, nu_init, Kd)
shift_phiRb = growth.model.phi_R_optimal_allocation(gamma_max,  nu_shift, Kd)
init_phiMb = 1 - init_phiRb
shift_phiMb = 1 - shift_phiRb


# set the initial conditions for the integration
M0 = 0.001 * OD_CONV
M_Rb = init_phiRb * M0
M_Mb = init_phiMb * M0
cpc = growth.model.steady_state_precursors(gamma_max, init_phiRb, nu_init, Kd)
init_params = [M_Rb, M_Mb, cpc]

# Set the two time ranges 
dt = 0.0001
preshift = np.arange(0, shift_time,dt)
postshift = np.arange(shift_time, 10, dt)

#
def integrate(params, t, gamma_max, nu_max, phiRb, phiMb, Kd=Kd):
    """
    Integrates the system of differential equations, including the dilution 
    factor and assumes that nutrient concentration is high enough such that 
    nu ≈ nu_max.
    """
    M_Rb, M_Mb, c_pc = params
    M = M_Rb +  M_Mb

    # Compute the elongation rate
    gamma = gamma_max * (c_pc / (c_pc + Kd))

    # Biomass dynamics
    dM_dt = gamma * M_Rb 

    # Precursor dynamics
    dc_pc_dt = nu_max * (M_Mb/M) - gamma * (M_Rb/M) * (1 + (c_pc/M))

    # Allocation
    dM_Rb_dt = phiRb * dM_dt
    dM_Mb_dt = phiMb * dM_dt
    return [dM_Rb_dt, dM_Mb_dt,  dc_pc_dt]

# Compute the constant scenario
const_preshift_args = (gamma_max, nu_init, init_phiRb, init_phiMb, Kd)
const_postshift_args = (gamma_max, nu_shift, init_phiRb, init_phiMb, Kd)
const_preshift_out = scipy.integrate.odeint(integrate, init_params,
                                            preshift, args=const_preshift_args)
preshift_df = pd.DataFrame(const_preshift_out, columns=['M_Rb', 'M_Mb', 'c_pc'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
const_postshift_params = const_preshift_out[-1]
const_postshift_out = scipy.integrate.odeint(integrate, const_postshift_params,
                                            postshift, args=const_postshift_args)
postshift_df = pd.DataFrame(const_postshift_out, columns=['M_Rb', 'M_Mb', 'c_pc'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] =  postshift

const_shift_df = pd.concat([preshift_df, postshift_df])
const_shift_df['total_biomass'] = const_shift_df['M_Rb'].values + const_shift_df['M_Mb'].values
const_shift_df['relative_biomass'] = const_shift_df['total_biomass'].values / M0
const_shift_df['prescribed_phiR'] = init_phiRb
const_shift_df['realized_phiR'] = const_shift_df['M_Rb'].values / const_shift_df['total_biomass'].values
const_shift_df['gamma'] = gamma_max * const_shift_df['c_pc'].values / (const_shift_df['c_pc'] + Kd)

# Optimal scenario
opt_preshift_args = (gamma_max, nu_init, init_phiRb, init_phiMb, Kd)
opt_postshift_args = (gamma_max, nu_shift, shift_phiRb, shift_phiMb, Kd)
opt_preshift_out = scipy.integrate.odeint(integrate, init_params,
                                            preshift, args=opt_preshift_args)
preshift_df = pd.DataFrame(const_preshift_out, columns=['M_Rb', 'M_Mb', 'c_pc'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
preshift_df['prescribed_phiR'] = init_phiRb
opt_postshift_params = opt_preshift_out[-1]
opt_postshift_out = scipy.integrate.odeint(integrate, opt_postshift_params,
                                            postshift, args=opt_postshift_args)
postshift_df = pd.DataFrame(opt_postshift_out, columns=['M_Rb', 'M_Mb', 'c_pc'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] =  postshift
postshift_df['prescribed_phiR'] = shift_phiRb

opt_shift_df = pd.concat([preshift_df, postshift_df])
opt_shift_df['total_biomass'] = opt_shift_df['M_Rb'].values + opt_shift_df['M_Mb'].values
opt_shift_df['relative_biomass'] = opt_shift_df['total_biomass'].values / M0
opt_shift_df['realized_phiR'] = opt_shift_df['M_Rb'].values / opt_shift_df['total_biomass'].values
opt_shift_df['gamma'] = gamma_max * opt_shift_df['c_pc'].values / (opt_shift_df['c_pc'] + Kd)


# Figure out where to start teh ppGpp model
init_params = [M_Rb, M_Mb, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, kappa_max)
postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, True, kappa_max)
out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 150, dt), args=preshift_args)
out = out[-1]
ppGpp_init_phiRb = (out[0]) / (out[0] + out[1])
ppGpp_init_phiMb = 1 - ppGpp_init_phiRb
init_T_AA = out[2]
init_T_AA_star = out[3]
ppGpp_init_params = [M0 * ppGpp_init_phiRb, M0 * ppGpp_init_phiMb, init_T_AA, init_T_AA_star]

# Compute the preshift ppGpp
ppGpp_preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                        ppGpp_init_params, preshift, args=preshift_args)
preshift_df = pd.DataFrame(ppGpp_preshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
preshift_df['nu'] = nu_init
preshift_df['phase'] = 'preshift'
preshift_df['time_hr'] =  preshift
ppGpp_shift_params = ppGpp_preshift_out[-1]
ppGpp_postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                        ppGpp_shift_params, postshift, args=postshift_args)
postshift_df = pd.DataFrame(ppGpp_postshift_out, columns=['M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
postshift_df['nu'] = nu_shift
postshift_df['phase'] = 'postshift'
postshift_df['time_hr'] = postshift 
ppGpp_shift_df = pd.concat([preshift_df, postshift_df])

# Compute properties
ppGpp_shift_df['total_biomass'] = ppGpp_shift_df['M_Rb'].values + ppGpp_shift_df['M_Mb'].values
ppGpp_shift_df['relative_biomass'] = ppGpp_shift_df['total_biomass'].values / M0
ppGpp_shift_df['tRNA_balance'] = ppGpp_shift_df['T_AA_star'].values / ppGpp_shift_df['T_AA'].values
ppGpp_shift_df['prescribed_phiR'] = ppGpp_shift_df['tRNA_balance'].values / (ppGpp_shift_df['tRNA_balance'].values + tau)
ppGpp_shift_df['realized_phiR'] = ppGpp_shift_df['M_Rb'].values / ppGpp_shift_df['total_biomass'].values
ppGpp_shift_df['gamma'] = gamma_max * ppGpp_shift_df['T_AA_star'].values / (ppGpp_shift_df['T_AA_star'].values + Kd_TAA_star)



# palette = sns.color_palette('crest', n_colors=len(nu_max) + 10)

# %%
fig, ax = plt.subplots(4, 1, figsize=(6, 6), sharex=True)
ax[0].set(ylabel='relative biomass', yscale='log')
ax[1].set(ylabel='ribosomal allocation $\phi_{Rb}$',
          ylim=[0, 1])
ax[2].set(ylabel='ribosome content $M_{Rb}/M$',
         ylim=[0, 1])
ax[3].set(ylabel='translation rate [AA / s]',
          xlabel='time [hr]',
          ylim=[0, 20])


# ppGpp model
ax[0].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['relative_biomass'], '-', 
            lw=1, label='dynamic re-allocation', color=colors['primary_red'])
ax[1].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['prescribed_phiR'], '-', lw=1, color=colors['primary_red'])
ax[2].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['realized_phiR'], '-', lw=1, color=colors['primary_red'])
ax[3].plot(ppGpp_shift_df['time_hr'], ppGpp_shift_df['gamma'] * 7459 / 3600, '-',  lw=1, color=colors['primary_red'])

# Constant model
ax[0].plot(const_shift_df['time_hr'], const_shift_df['relative_biomass'], 'k-', 
            lw=1, label='fixed allocation')
ax[1].plot(const_shift_df['time_hr'], const_shift_df['prescribed_phiR'], 'k-', lw=1)
ax[2].plot(const_shift_df['time_hr'], const_shift_df['realized_phiR'], 'k-', lw=1)
ax[3].plot(const_shift_df['time_hr'], const_shift_df['gamma'] * 7459 / 3600, 'k-',  lw=1)

# Optimal model
ax[0].plot(opt_shift_df['time_hr'], opt_shift_df['relative_biomass'], '-', 
            lw=1, label='instantaneous  re-allocation', color=colors['primary_blue'])
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
plt.savefig('../figures/ppGpp_shift_strategies.pdf', bbox_inches='tight')
# %%
