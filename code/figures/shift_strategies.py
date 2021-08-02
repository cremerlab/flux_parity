#%%
import numpy as np 
import pandas as pd 
import tqdm
import growth.viz 
import growth.model
import matplotlib.pyplot as plt
colors, palette = growth.viz.matplotlib_style()


#%%
# Define constants for the integration 
OD_CONV = 1.5E17
gamma_max = 9.65
# Approximate Kd given intracellular amino acid concentrations
Kd = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23)

# Define the shift (either up- or down-shift)
nu_init = 8 
nu_shift = 2 
time_shift = 0.5 # Time point at which the shift occurs.
 
# Set the initial conditions
M0 = 0.1 * OD_CONV
phiR_init = growth.model.phi_R_optimal_allocation(gamma_max, nu_init, Kd, phi_O=0)
# Above lines comes out to ≈ 0.1

cAA_init = growth.model.sstRNA_balance(nu_init, 1- phiR_init, gamma_max, phiR_init, Kd)
# above line ocmes out to ≈ 0.05

Mr_init = phiR_init * M0
Mp_init = (1 - phiR_init) * M0

# Define the time ranges
T_START = 0 
T_END =  1 
N_STEPS = T_END * 60  * 60 # steps of 1 s
dt = (T_END - T_START) / N_STEPS
time_range = np.linspace(T_START, T_END, N_STEPS)

# Set the range of ribosomal allocation parameters to consider.
phiR_range = np.linspace(0, 1, 500)

def integrate(params, args, Kd=Kd, dt=dt):
    """
    Integrates the system of differential equations, including the dilution 
    factor and assumes that nutrient concentration is high enough such that 
    nu ≈ nu_max.
    """
    # Unpacking
    M, Mr, Mp, cAA = params
    gamma_max, nu_max, phiR = args

    # Translational efficiency
    gamma = gamma_max * (cAA / (cAA + Kd))

    # Biomass dynamics
    dM_dt = gamma * Mr
            
    # Metabolism
    dcAA_dt = (nu_max * Mp - (1 + cAA) * dM_dt) / M
 
    # Allocation
    dMr_dt = phiR * dM_dt 
    dMp_dt = (1 - phiR) * dM_dt
    
    return np.array([dM_dt, dMr_dt, dMp_dt, dcAA_dt]) * dt

#%%
dfs = []
for i, strat in enumerate(tqdm.tqdm(['constant', 'optimal', 'elongation'])): 

    # Set the output vector and the initial conditions
    out = np.zeros((4, len(time_range)))
    out[:, 0] = [M0, Mr_init, Mp_init, cAA_init]
    phiR_val = np.zeros_like(time_range)
    phiR_val[0] = phiR_init

    # Loop through the time step
    for j in range(1, len(time_range)):
        # Set the initial conditions for the time step given the system configuration
        # at the previous time step
        params = out[:, j-1]

        # If in the preshift condition, integrate given the initial steady state
        if time_range[j] < time_shift:
            args = (gamma_max, nu_init, phiR_init)
            _out = integrate(params, args) 
            out[:, j] = out[:, j-1] + _out
            phiR_val[j] = phiR_init

        # Otherwise, shift the the value of nu and choose an allocation strategy
        else:
            # Strategy 1: Constant allocation, keep everything the same as the 
            # initialization, save for a change in nu
            if strat == 'constant':
                args = (gamma_max, nu_shift, phiR_init)
                _out = integrate(params, args)
                out[:, j] = out[:, j-1] + _out
                phiR_val[j] = phiR_init 
            # Strategy 2 & 3: Figure out the right phiR to choose.  
            else:  
                # Set a storage vector for the changes in biomass and cAA as 
                # phiR is tuned.
                biomass_diff = np.zeros_like(phiR_range)
                cAA = np.zeros_like(phiR_range)

                # Sweep over values of phiR.
                for k, phi in enumerate(phiR_range):
                    args = (gamma_max, nu_shift,  phi)

                    # FIrst integration uses current Mr and Mp, so need to 
                    # integrate two timesteps to actually see the effect of a 
                    # changing phiR
                    _params = integrate(params, args)
                    _params += out[:, j-1]
                    _out = integrate(_params, args) 

                    # Save the change in the biomass and in the cAA.
                    biomass_diff[k] = _out[0]
                    cAA[k] = _params[-1] + _out[-1]

                # Given cAA, compute the translation rate
                gamma = gamma_max * (cAA / (cAA + Kd))
                # Strategy 2: 'Optimal' allocation to maximize growth rate. 
                # Choose the value of phiR which leads to the largest value of 
                # dM_dt
                if strat == 'optimal':
                    ind = np.argmax(biomass_diff)
                    phi = phiR_range[ind]
                    args= (gamma_max, nu_shift, phi)
                    _out = integrate(params, args) 
                    out[:, j] = out[:, j-1] + _out
                    phiR_val[j] = phi
                
                # Strategy 3: Restrain the ribosomal allocation such that 
                # translation rate is maximized

                elif strat == 'elongation':
                    ind = np.argmax(gamma)
                    phi = phiR_range[ind]
                    args= (gamma_max, nu_shift, phi)
                    _out = integrate(params, args) 
                    phiR_val[j] = phi
                    out[:, j] = out[:, j-1] + _out

    # Store everything as a dataframe
    df = pd.DataFrame(out.T, columns=['M', 'Mr', 'Mp', 'cAA'])
    df['phiR'] = phiR_val
    df['rel_biomass'] = df['M'] / M0
    df['strategy'] = strat
    df['time'] = time_range
    dfs.append(df)

#%%
fig, ax = plt.subplots(3, 1, figsize=(6, 4))
#%%
alt.data_transformers.disable_max_rows()


# Merge everything into a single dataframe
df = pd.concat(dfs, sort=False)
alt.Chart(df).mark_line().encode(
        x=alt.X('time:Q'),
        y=alt.Y('rel_biomass:Q', scale=alt.Scale(type='log')),
        color=alt.Color('strategy:N'),
        strokeDash=alt.StrokeDash('strategy:N')
).interactive()

# %%

phiR_plot = alt.Chart(df).mark_line().encode(
        x=alt.X('time:Q'),
        y=alt.Y('phiR:Q'),
        color=alt.Color('strategy:N'),
        strokeDash=alt.StrokeDash('strategy:N')
)

cAA_plot = alt.Chart(df).mark_line().encode(
        x=alt.X('time:Q'),
        y=alt.Y('cAA:Q'),
        color=alt.Color('strategy:N'),
        strokeDash=alt.StrokeDash('strategy:N')
)
phiR_plot & cAA_plot

# %%

# %%
