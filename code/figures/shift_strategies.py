#%%
import numpy as np 
import pandas as pd 
import tqdm
import growth.viz 
import growth.model
import scipy.integrate
import altair as alt
import altair_saver
colors, palette = growth.viz.altair_style()
alt.data_transformers.disable_max_rows()

#%%
# Define constants for the integration 
OD_CONV = 1.5E17
gamma_max = 9.65
kappa = 0.1
# Approximate Kd given intracellular amino acid concentrations
Kd = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23)

# Define the shift (either up- or down-shift)
nu_init = 5 
nu_shift =  1 
time_shift = 0.5 # Time point at which the shift occurs.
 
# Set the initial conditions
M0 = 0.04 * OD_CONV
phiR_init = growth.model.phi_R_optimal_allocation(gamma_max, nu_init, Kd, phi_O=0)
cAA_init = growth.model.sstRNA_balance(nu_init, 1- phiR_init, gamma_max, phiR_init, Kd)
gamma_init = gamma_max * (cAA_init / (cAA_init + Kd))
Mr_init = phiR_init * M0
Mp_init = (1 - phiR_init) * M0

# Define the values for the "optimal" solution.
phiR_final_optimal = growth.model.phi_R_optimal_allocation(gamma_max, nu_shift, Kd, phi_O=0)

# Define the values for the "constant" solution
phiR_final_constant = phiR_init

# Define the values for the maintenance of cAA
gamma = gamma_max * cAA_init / (cAA_init + Kd)
phiR_final_elong =  nu_shift / (gamma * (cAA_init + 1) + nu_shift) 

# Define the time ranges
T_START = 0 
T_END =  5 

# N_STEPS = T_END * 3600
N_STEPS = 2000 
dt = (T_END - T_START) / N_STEPS
time_range = np.linspace(T_START, T_END, N_STEPS)

# Set the range of ribosomal allocation parameters to consider.
phiR_range = np.linspace(0, 1, 500)

def integrate(params, t, gamma_max, nu_max, phiR, Kd=Kd):
    """
    Integrates the system of differential equations, including the dilution 
    factor and assumes that nutrient concentration is high enough such that 
    nu ≈ nu_max.
    """
    Mr, Mp, Mr_star, cAA = vars
    M = Mr + Mr_star + Mp

    # Compute the elongation rate
    gamma = gamma_max * (cAA / (cAA + Kd))

    # Biomass dynamics
    dM_dt = gamma * Mr 

    # Precursor dynamics
    dcAA_dt = nu_max * (Mp/M) - gamma * (Mr/M) * (1 + (cAA/M))

    # Allocation
    dMr_dt = kappa * Mr_star
    dMr_star_dt = phiR * dM_dt - kappa * Mr_star
    dMp_dt = (1 - phiR) * dM_dt
    return [dMr_dt, dMp_dt, dMr_star_dt, dcAA_dt]


    # # Unpacking
    # Mr, Mp, cAA = params
    # gamma_max, nu_max, phiR = args

    # # Translational efficiency
    # gamma = gamma_max * (cAA / (cAA + Kd))

    # # Biomass dynamics
    # dM_dt = gamma * Mr
            
    # # Metabolism
    # dcAA_dt = (nu_max * Mp - (1 + cAA) * dM_dt) / (Mr + Mp)

    # # Allocation
    # dMr_dt = phiR * dM_dt 
    # dMp_dt = (1 - phiR) * dM_dt    
    # return np.array([dMr_dt, dMp_dt, dcAA_dt])

# #%% Scenario I: Instantaneous changing of phiR
# dfs = []
# for i, strat in enumerate(tqdm.tqdm(['constant', 'optimal', 'elongation'])):
#     # Set the output vector and the initial conditions
#     out = np.zeros((3, len(time_range)))
#     out[:, 0] = [Mr_init, Mp_init, cAA_init]
#     nu = np.zeros_like(time_range)
#     phi = np.zeros_like(time_range)
#     nu[0] = nu_init
#     phi[0] = phiR_init
    
#     # Loop through the time step
#     for j in range(1, len(time_range)):
#         # Set the initial conditions for the time step given the system configuration
#         # at the previous time step
#         params = out[:, j-1]

#         # If in the preshift condition, integrate given the initial steady state
#         if time_range[j] < time_shift:
#             _nu = nu_init
#             _phi = phiR_init
#         else:
#             _nu = nu_shift 
#             if strat == 'constant':
#                 _phi = phiR_final_constant  
#             elif strat == 'optimal':
#                 _phi = phiR_final_optimal 
#             elif strat == 'elongation':
#                 _phi = phiR_final_elong

#         args = (gamma_max, _nu, _phi)
#         _out = scipy.integrate.odeint(integrate, params, [0, dt], args=args)
#         out[:, j] = _out[-1]
#         nu[j] = _nu
#         phi[j]= _phi

#     # Store everything as a dataframe
#     df = pd.DataFrame(out.T, columns=['Mr', 'Mp', 'cAA'])
#     df['rel_biomass'] = (df['Mr'].values + df['Mp'].values )/ M0
#     df['strategy'] = strat
#     df['time'] = time_range
#     df['nu'] = nu
#     df['phiR'] = df['Mr'].values / (df['Mr'].values + df['Mp'].values)
#     dfs.append(df)

# instantaneous_df = pd.concat(dfs, sort=False)

# #%%
# # Set up the plots
# base = alt.Chart(instantaneous_df)
# nu_plot =  base.mark_line().encode(
#                 x=alt.X('time:Q', title='time [hr]'),
#                 y=alt.Y('nu:Q', title='ν [per hr]')
#             ).properties(width=800, height=100)

# M_plot = base.mark_line().encode(
#             x=alt.X('time:Q', title='time [hr]'),
#             y=alt.Y('rel_biomass:Q', title='relative biomass', 
#                     # scale=alt.Scale(type='log')
#                     ),
#             color=alt.Color('strategy:N'),
#             strokeDash=alt.StrokeDash('strategy:N')
#             ).properties(width=250, height=250)
# cAA_plot = base.mark_line().encode(
#             x=alt.X('time:Q', title='time [hr]'),
#             y=alt.Y('cAA:Q', title='precursor abundance'),
#             color=alt.Color('strategy:N'),
#             strokeDash=alt.StrokeDash('strategy:N')
#             ).properties(width=250, height=250)
# phiR_plot = base.mark_line().encode(
#             x=alt.X('time:Q', title='time [hr]'),
#             y=alt.Y('phiR:Q', title='ribosome allocation ΦR'),
#             color=alt.Color('strategy:N'),
#             strokeDash=alt.StrokeDash('strategy:N')
#             ).properties(width=250, height=250)


# layout = nu_plot & (M_plot  | cAA_plot | phiR_plot)
# # altair_saver.save(layout, './nutrient_shift_instantaneous_reallocation.pdf')
# layout
#%% Scenario II: Dynamic changing of phiR
dfs = []
for i, strat in enumerate(tqdm.tqdm(['constant', 'optimal', 'elongation'])): 

    # Set the output vector and the initial conditions
    out = np.zeros((3, len(time_range)))
    out[:, 0] = [Mr_init, Mp_init, cAA_init]
    phi = np.zeros_like(time_range)
    nu = np.zeros_like(time_range)
    nu[0] = nu_init
    phi[0] = phiR_init
    
    # Loop through the time step
    for j in tqdm.tqdm(range(1, len(time_range)), desc='Iterating over time points...'):
        # Set the initial conditions for the time step given the system configuration
        # at the previous time step
        params = out[:, j-1]

        # If in the preshift condition, integrate given the initial steady state
        if time_range[j] < time_shift:
            args = (gamma_max, nu_init, phiR_init)
            _out = scipy.integrate.odeint(integrate, params, [0, dt], args=args) 
            out[:, j] = _out[-1]
            phi[j] = phiR_init
            nu[j] = nu_init

        # Otherwise, shift the the value of nu and choose an allocation strategy
        else:
            nu[j] = nu_shift
            # Strategy 1: Constant allocation, keep everything the same as the 
            # initialization, save for a change in nu
            if strat == 'constant':
                args = (gamma_max, nu_shift, phiR_init)
                _out = scipy.integrate.odeint(integrate, params, [0, dt], args=args)
                out[:, j] = _out[-1]
                phi[j] = phiR_init 
            # Strategy 2 & 3: Figure out the right phiR to choose.  
            else:  
                # Set a storage vector for the changes in biomass and cAA as 
                # phiR is tuned.
                biomass = np.zeros_like(phiR_range)
                cAA = np.zeros_like(phiR_range)

                # Sweep over values of phiR.
                _outs = []
                for k, _phi in enumerate(phiR_range):
                    args = (gamma_max, nu_shift, _phi)
                    _out = scipy.integrate.odeint(integrate, params, [0, dt, 0.05], args=args)

                    # Save the change in the biomass and in the cAA.
                    _out = _out
                    _outs.append(_out[-2])
                    biomass[k] = _out[-1][0] + _out[-1][1]
                    cAA[k] = _out[-1][-1]
                # Strategy 2: 'Optimal' allocation to maximize growth rate. 
                # Choose the value of phiR which leads to the largest value of 
                # dM_dt
                if strat == 'optimal':
                    ind = np.argmax(biomass)
                    _phi = phiR_range[ind]
                    out[:, j] =  _outs[ind]
                    phi[j] = _phi
                
                #Strategy 3: Restrain the ribosomal allocation such that 
                # translation rate is kept the same

                elif strat == 'elongation':
                    ind = np.where(abs(cAA - cAA_init) < 0.001)[0]
                    if len(ind) >= 1:
                        ind = ind[0]
                    elif len(ind) == 0:
                        if np.min(cAA) > cAA_init:
                            ind = len(phiR_range) -1
                        else:
                            ind = 0 
                    _phi = phiR_range[ind]  
                    phi[j] = _phi
                    out[:, j] = _outs[ind]

    # Store everything as a dataframe
    df = pd.DataFrame(out.T, columns=['Mr', 'Mp', 'cAA'])
    df['phiR'] = phi
    df['nu'] = nu
    df['rel_biomass'] = (df['Mr'].values + df['Mp'].values) / M0
    df['strategy'] = strat
    df['time'] = time_range
    dfs.append(df)
dynamic_df = pd.concat(dfs,sort=False)
#%%

# Set up the plots
base = alt.Chart(dynamic_df)
nu_plot =  base.mark_line().encode(
                x=alt.X('time:Q', title='time [hr]'),
                y=alt.Y('nu:Q', title='ν [per hr]')
            ).properties(width=800, height=100)
M_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('rel_biomass:Q', title='relative biomass', 
                    scale=alt.Scale(type='log')),
            color=alt.Color('strategy:N'),
            strokeDash=alt.StrokeDash('strategy:N')
            ).properties(width=250, height=250)
cAA_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('cAA:Q', title='precursor abundance'),
            color=alt.Color('strategy:N'),
            strokeDash=alt.StrokeDash('strategy:N')
            ).properties(width=250, height=250)
phiR_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('phiR:Q', title='ribosome allocation ΦR'),
            color=alt.Color('strategy:N'),
            strokeDash=alt.StrokeDash('strategy:N')
            ).properties(width=250, height=250)


layout = nu_plot & (M_plot  | cAA_plot | phiR_plot)
# altair_saver.save(layout, f'./nutrient_upshift_dynamic_reallocation_{N_STEPS}timesteps.pdf')
layout


# %%

# %%
