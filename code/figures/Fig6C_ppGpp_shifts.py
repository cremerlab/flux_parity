#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.viz
import tqdm
import scipy.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Set the constants
gamma_max = const['gamma_max']
nu_init = [5, 10]
nu_shift = [10, 5]
nu_max = [nu_init, nu_shift]
prefactors = [[1, 0], [0, 1]]
total_time = 5 
shift_time = 1
phi_O = 0.55
phi_Mb_0 = 0.1
dt = 0.0001
preshift_time = np.arange(0, shift_time, dt)
postshift_time = np.arange(shift_time-dt, total_time, dt)

# ppGpp params
Kd_TAA = 2E-5 #const['Kd_TAA']
Kd_TAA_star = 2E-5 # const['Kd_TAA_star']
tau =  1 #const['tau'] 
phi_Rb = 0.1
phi_Mb = 1 - phi_O - phi_Rb

# Define the integration function with multiple metabolic casettes
def ppGpp_shift(params, 
                t,
                gamma_max,
                nu_max, 
                tau, 
                Kd_TAA,
                Kd_TAA_star,
                kappa_max,
                phi_O,
                phi_Mb_0,
                prefactors,
                dynamic_phiRb = True,
                phiRb = 0):
    M, M_Rb, M_Mb_1, M_Mb_2, TAA, TAA_star = params

    # Define rates
    gamma = gamma_max * (TAA_star / (TAA_star + Kd_TAA_star))
    nu_1 = prefactors[0] * nu_max[0] * (TAA / (TAA + Kd_TAA))
    nu_2 = prefactors[1] * nu_max[1] * (TAA / (TAA + Kd_TAA))

    # Define phiRb
    if dynamic_phiRb: 
        ratio = TAA_star / TAA
        phiRb = (1 - phi_O) * ratio / (ratio + tau)
        kappa = kappa_max  * ratio / (ratio + tau)
    else:
        kappa = kappa_max * phi_Rb
    # Encode dynamics
    dM_dt = gamma * M_Rb
    dTAA_star_dt = (nu_1 * M_Mb_1 + nu_2 * M_Mb_2 - dM_dt * (1 + TAA_star)) / M
    dTAA_dt = kappa + (dM_dt * (1 - TAA) - nu_1 * M_Mb_1 - nu_2 * M_Mb_2) / M

    # Encode allocation
    dM_Rb_dt = phiRb * dM_dt
    if prefactors[0] == 1:
        dM_Mb_2_dt = phi_Mb_0 * dM_dt
        dM_Mb_1_dt = (1 - phi_O - phiRb - phi_Mb_0) * dM_dt
    else:
        dM_Mb_1_dt = phi_Mb_0 * dM_dt
        dM_Mb_2_dt = (1 - phi_O - phi_Rb - phi_Mb_0) * dM_dt

    return [dM_dt, dM_Rb_dt, dM_Mb_1_dt, dM_Mb_2_dt, dTAA_dt, dTAA_star_dt]


# Init params
M0 = 0.03 * const['OD_conv']
M_Rb = phi_Rb * M0
M_Mb = (1 - phi_Rb - phi_O) * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = const['kappa_max']
dt = 0.0001
time = np.arange(0, 10, dt)

dynamic_dfs = []
instant_dfs = []
for  i, nu in enumerate(tqdm.tqdm(nu_max)):
        nu_pre, nu_post = nu
        # Determine the type of shift
        if nu_post > nu_pre:
            shift_type = 'upshift'
            mult_pre = 1.2
            mult_post = 1
        else:
            shift_type = 'downshift'
            mult_pre = 1
            mult_post = 1.2
 
        init_params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
        preshift_args = (gamma_max, nu[0], tau, Kd_TAA, Kd_TAA_star,
                        kappa_max, phi_O * mult_pre)
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                             init_params,np.arange(0, 200, dt), args=preshift_args)
        postshift_args = (gamma_max, nu[1], tau, Kd_TAA, Kd_TAA_star, 
                          kappa_max, phi_O * mult_post)
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=postshift_args)


        # Compute the initial states
        preshift_out = preshift_out[-1]
        postshift_out = postshift_out[-1]
        init_phiRb = preshift_out[1] / preshift_out[0]
        init_phiMb = preshift_out[2] / preshift_out[0]
        shift_phiRb = postshift_out[1] / postshift_out[0]
        shift_phiMb = postshift_out[2] / postshift_out[0]
        init_T_AA = preshift_out[-2]
        init_T_AA_star = preshift_out[-1]

        # Perform the shift for dynamic reallocation
        init_params = [M0, M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
        preshift_args = (gamma_max, nu[0], tau, Kd_TAA, Kd_TAA_star, 
                        kappa_max, phi_O * mult_pre)
        postshift_args = (gamma_max, nu[1], tau, Kd_TAA, Kd_TAA_star,
                          kappa_max, phi_O * mult_post)

        # Integrate preshift
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, preshift_time, args=preshift_args)
        shift_params = preshift_out[-1]
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                shift_params, postshift_time, args=postshift_args)
        postshift_out = postshift_out[1:]


        # Generate dataframes and concat
        preshift_df = pd.DataFrame(preshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        postshift_df = pd.DataFrame(postshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        preshift_df['time'] = preshift_time
        postshift_df['time'] = postshift_time[1:]
        dynamic_df = pd.concat([preshift_df, postshift_df], sort=False)
        dynamic_df['balance'] = dynamic_df['TAA_star'].values / dynamic_df['TAA'].values
        dynamic_df['phi_Rb'] = (1 - phi_O) * dynamic_df['balance'].values / (dynamic_df['balance'].values + tau) 
        dynamic_df['MRb_M'] = dynamic_df['MRb'].values / dynamic_df['M'].values
        dynamic_df['shift_type'] = shift_type
        dynamic_dfs.append(dynamic_df) 

        # Perform the shift for instant reallocation
        init_params = [M0, M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
        preshift_args = (gamma_max, nu[0], tau, Kd_TAA, Kd_TAA_star, 
                        kappa_max, phi_O * mult_pre, False, False, False, True, init_phiRb)
        postshift_args = (gamma_max, nu[1], tau, Kd_TAA, Kd_TAA_star,
                          kappa_max, phi_O * mult_post, False, False, False, True, shift_phiRb)

        # Integrate preshift
        preshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                init_params, preshift_time, args=preshift_args)
        shift_params = preshift_out[-1]
        postshift_out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                                shift_params, postshift_time, args=postshift_args)
        postshift_out = postshift_out[1:]

        # Generate dataframes and concat
        preshift_df = pd.DataFrame(preshift_out, columns=['M', 'MRb', 'Mmb', 'TAA', 'TAA_star'])
        postshift_df = pd.DataFrame(postshift_out, columns=['M', 'MRb', 'Mmb','TAA', 'TAA_star'])
        preshift_df['time'] = preshift_time
        preshift_df['phi_Rb'] = init_phiRb
        postshift_df['time'] = postshift_time[1:]
        postshift_df['phi_Rb'] = shift_phiRb
        instant_df = pd.concat([preshift_df, postshift_df], sort=False)
        instant_df['MRb_M'] = instant_df['MRb'].values / instant_df['M'].values
        instant_df['balance'] = instant_df['TAA_star'].values / instant_df['TAA'].values
        instant_df['shift_type'] = shift_type
        instant_dfs.append(instant_df)

dynamic_df = pd.concat(dynamic_dfs, sort=False)
dynamic_df['time'] -= shift_time
instant_df = pd.concat(instant_dfs, sort=False)
instant_df['time'] -= shift_time
# %%

# Make plot first for only the upshift
dynamic_up = dynamic_df[dynamic_df['shift_type']=='upshift']
instant_up = instant_df[instant_df['shift_type']=='upshift']

fig, ax = plt.subplots(1, 3, figsize=(6, 2))

# Format and label axes
for a in ax:
    a.set_xlim([-1, 8])
ax[0].set_yscale('log')
ax[0].set_ylim([1E-2, 5])
ax[1].set_ylim([0.1, 0.5])
ax[2].set_ylim([0.8, 2])
for a in ax:
    a.set_xlabel('time from upshift [hr]')

ax[0].set_ylabel('approximate\n optical density [a.u.]')
ax[1].set_ylabel('$\phi_{Rb}$\nallocation to ribosomes')
ax[2].set_ylabel('$\lambda_{instant}$ [hr$^{-1}$]\ninstantaneous growth rate')

# Instant
ax[0].plot(instant_up['time'], instant_up['M'].values / const['OD_conv'], '-', color=colors['primary_blue'], lw=1)
ax[1].plot(instant_up['time'], instant_up['phi_Rb'], '-', color=colors['primary_blue'], lw=1)
instant_gr = np.log(instant_up['M'].values[1:] / instant_up['M'].values[:-1]) / dt
ax[2].plot(instant_up['time'].values[1:], instant_gr, '-', color=colors['primary_blue'], lw=1)

# Dynamic
ax[0].plot(dynamic_up['time'], dynamic_up['M'].values / const['OD_conv'], '--', color=colors['primary_red'], lw=1)
ax[1].plot(dynamic_up['time'], dynamic_up['phi_Rb'], '--', color=colors['primary_red'], lw=1)
dynamic_gr = np.log(dynamic_up['M'].values[1:] / dynamic_up['M'].values[:-1]) / dt
ax[2].plot(dynamic_up['time'].values[1:], dynamic_gr, '--', color=colors['primary_red'], lw=1)
plt.tight_layout()
plt.savefig('../../figures/Fig6C_upshift_plots.pdf', bbox_inches='tight')
#%%
fig, ax = plt.subplots(1, 4, figsize=(7, 2), sharex=True)
ax = np.array([[ax[0],ax[1]], [ax[2], ax[3]]])


# Add labels
for i in range(2):
    ax[i, 0].set_xlabel('time from upshift [hr]')
    ax[i, 1].set_xlabel('time from downshift [hr]')

for i in range(2):
    ax[0, i].set_ylabel('$\phi_{Rb}$\nallocation towards ribosomes')
    ax[0,i].set_ylabel('$\lambda$ [hr$^{-1}$]\ninstantaneous growth rate')

# Set limits
ax[0, 0].set_ylim([0, 0.8])
ax[0, 1].set_ylim([0, 0.8])

for g, d in instant_df.groupby(['shift_type']):
    if g == 'upshift':
        ind = 0
    else:
       ind = 1 
    ax[0, ind].plot(d['time'], d['phi_Rb'], '-', color=colors['primary_blue'], lw=1)
    gr = np.log(d['M'].values[1:] / d['M'].values[:-1]) / dt
    ax[1, ind].plot(d['time'].values[1:], gr, '-', color=colors['primary_blue'], lw=1)


for g, d in dynamic_df.groupby(['shift_type']):
    if g == 'upshift':
        ind = 0
    else:
        ind = 1
    ax[0, ind].plot(d['time'], d['phi_Rb'], '--', color=colors['primary_red'], lw=1)
    gr = np.log(d['M'].values[1:] / d['M'].values[:-1]) / dt
    ax[1, ind].plot(d['time'].values[1:], gr, '--', color=colors['primary_red'], lw=1)

plt.tight_layout()
plt.subplots_adjust(hspace=0.05)
plt.savefig('../../figures/Fig6C_ppGpp_shift_plots.pdf', bbox_inches='tight')
# %%


# %%
