#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import scipy.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

def shift(params,
          time,
          gamma_max,
          nu_max, 
          tau, 
          Kd_TAA,
          Kd_TAA_star,
          kappa_max,
          phi_O,
          prefactors,
          phi_Rb = 0.1,
          dil_approx = False,
          dynamic_phiRb = True,
          tRNA_regulation = True
          ):

    # Unpack the parameters
    M, M_Rb, M_Mb_1, M_Mb_2, T_AA, T_AA_star = params

    # Compute the capacities
    gamma = gamma_max * (T_AA_star / (T_AA_star + Kd_TAA_star))
    nu_1 = nu_max[0] * (T_AA / (T_AA + Kd_TAA))
    nu_2 = nu_max[1] * (T_AA / (T_AA + Kd_TAA))

    # Compute the active fraction
    ratio = T_AA_star / T_AA

    # Biomass accumulation
    dM_dt = gamma * M_Rb

    # Resource allocation
    if dynamic_phiRb:
        phi_Rb = ratio / (ratio + tau)

    dM_Rb_dt = phi_Rb * dM_dt
    dM_Mb1_dt = prefactors[0] * (1 - phi_Rb - phi_O) * dM_dt
    dM_Mb2_dt = prefactors[1] * (1 - phi_Rb - phi_O) * dM_dt

    # tRNA dynamics
    dT_AA_star_dt = (prefactors[0] * nu_1 * M_Mb_1  + prefactors[1] * nu_2 * M_Mb_2 - dM_dt) / M
    dT_AA_dt = (dM_dt - prefactors[0] * nu_1 * M_Mb_1 - prefactors[1] * nu_2 * M_Mb_2) / M
    if dil_approx == False:
        dT_AA_star_dt -= T_AA_star * dM_dt / M
        if tRNA_regulation:
            kappa = kappa_max * phi_Rb
        else:
            kappa = kappa_max
        dT_AA_dt += kappa - (T_AA * dM_dt) / M

    # Pack and return the output.
    out = [dM_dt, dM_Rb_dt, dM_Mb1_dt, dM_Mb2_dt, dT_AA_dt, dT_AA_star_dt]
    return out

gamma_max = const['gamma_max']
phi_O = 0.3
Kd = 2E-5
tau = const['tau']
kappa_max = const['kappa_max']
prefactors = [[1, 0], [0, 1]]
nu_max_1 = 2 
nu_max_2 = 0.5 
M0 = 1E9

# Find the equilibrium from the ppGpp model
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max_1, 0.01, phi_O)
params = [M0, opt_phiRb * M0, (1 - opt_phiRb - phi_O) * M0, 0, 0.002, 0.002]
args = (gamma_max, [nu_max_1, nu_max_2], tau, Kd, Kd, kappa_max, phi_O, [1, 0])
dt = 0.0001
t_range = np.arange(0, 200, dt)
out = scipy.integrate.odeint(shift, params, t_range, args=args)
out = out[-1]
denom = 10**np.log10(out[0])
_out = [out[0] / denom, out[1] / denom, out[2] / denom, out[3]/denom, out[4], out[5]]

# Set the initial conditions for the shift
opt_phiRb = out[1] / out[0]
M0 = 0.0089 * const['OD_conv'] 
MRb = opt_phiRb * M0
MMb_1 = (1 - phi_O - opt_phiRb) * M0
MMb_2 = 0
TAA = out[-2]
TAA_star = out[-1]

# Define the two time ranges
t_shift = 2 
dt = 0.001
t_end = 50 
preshift_time = np.arange(0, t_shift + dt, dt)
postshift_time = np.arange(t_shift, t_end, dt)

# Integrate the preshift
init_params = [M0, MRb, MMb_1, MMb_2, TAA, TAA_star]
init_args = (gamma_max, [nu_max_1, nu_max_2], tau, Kd, Kd, kappa_max, phi_O, [1, 0])
preshift_int = scipy.integrate.odeint(shift, init_params, preshift_time, args=init_args)

# Integrate the postshift
shift_params = preshift_int[-1]
shift_args = (gamma_max, [nu_max_1, nu_max_2], tau, Kd, Kd, kappa_max, phi_O, prefactors[1])
postshift_int = scipy.integrate.odeint(shift, shift_params, postshift_time - t_shift, args=shift_args)
postshift_int = postshift_int[1:]

# Set up the dataframe
preshift_df = pd.DataFrame(preshift_int, columns=['M', 'MRb', 'MMb_1', 'MMb_2', 'TAA', 'TAA_star'])
preshift_df['time'] = preshift_time
postshift_df = pd.DataFrame(postshift_int, columns=['M', 'MRb', 'MMb_1', 'MMb_2', 'TAA', 'TAA_star'])
postshift_df['time'] = postshift_time[1:]

# Make a single dataframe and compute the instantaneous gr
shift_df = pd.concat([preshift_df, postshift_df])
shift_df['balance'] = shift_df['TAA_star'].values / shift_df['TAA']
shift_df['phi_Rb'] = shift_df['balance'].values / (shift_df['balance'].values + tau)
shift_df['phi_Mb'] = 1 - phi_O - shift_df['phi_Rb']
shift_df['MRbM'] = shift_df['MRb'].values / shift_df['M']
shift_df['MMb_1_M'] = shift_df['MMb_1'].values / shift_df['M']
shift_df['MMb_2_M'] = shift_df['MMb_2'].values / shift_df['M']
shift_df['time'] -= t_shift

gr = list(np.log(shift_df['M'].values[1:]/ shift_df['M'].values[:-1]) / dt)
gr.append(gr[-1])
shift_df['inst_gr'] = gr


fig, ax = plt.subplots(2, 4, figsize=(8, 4))
ax = ax.ravel()
ax[0].set_yscale('log')
ax[0].set_ylim([1E-3, 0.5])
ax[0].set_xlim([-2, 10])
ax[2].set_ylim([-0.1, 1.1])

ax[3].set_ylim([-0.1, 1.1])
ax[4].set_ylim([-0.1, 1.1])
ax[5].set_ylim([-0.1, 1.1])
ax[6].set_ylim([-0.1, 1.1])
for a in ax:
    a.set_xlabel('time from shift [hr]')

ax[0].set_ylabel('relative biomass $M / M_0$')
ax[1].set_ylabel('instantaneous growth rate $\lambda$ [hr$^{-1}$]')
ax[2].set_ylabel('$\phi_{Rb}$')
ax[3].set_ylabel('$\phi_{Mb}$')
ax[4].set_ylabel('$M_{Rb} / M$')
ax[5].set_ylabel('$M_{Mb, 1} / M$')
ax[6].set_ylabel('$M_{Mb, 2} / M$')

ax[0].plot(shift_df['time'], shift_df['M'].values / const['OD_conv'], '-', color=colors['light_blue'], lw=1, zorder=100)
ax[1].plot(shift_df['time'], shift_df['inst_gr'], '-', color=colors['light_blue'], lw=1)
ax[2].plot(shift_df['time'], shift_df['phi_Rb'].values, '-', color=colors['light_blue'], lw=1)
ax[3].plot(shift_df['time'], shift_df['phi_Mb'].values, '-', color=colors['light_blue'], lw=1)

ax[4].plot(shift_df['time'], shift_df['MRbM'], '-', color=colors['light_blue'], lw=1)
ax[5].plot(shift_df['time'], shift_df['MMb_1_M'], '-', color=colors['light_blue'], lw=1)
ax[6].plot(shift_df['time'], shift_df['MMb_2_M'], '-', color=colors['light_blue'], lw=1)
ax[7].set_visible(False)

plt.tight_layout()
plt.savefig('./ppGpp_downshift_plots.pdf', bbox_inches='tight')
# %%

# %%
