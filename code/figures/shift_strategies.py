#%%
import numpy as np 
import pandas as pd 
import altair as alt 
import tqdm
import growth.viz 
import growth.model
colors, palette = growth.viz.altair_style()

# Set the initial conditions

VOL = 1E-3
OD_CONV = 1.5E17
gamma_max = 9.65
Kd = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23)
M0 = 0.1 * OD_CONV
nu_init = 2
nu_shift = 4 
time_shift = 1

# Define the nutrients and yield
nutrient_mw = 180 * (6.022E23 / 110)
omega = 0.3 * VOL * nutrient_mw

# Set the phiR 
phiR_init = growth.model.phi_R_optimal_allocation(gamma_max, nu_init, Kd, phi_O=0)
phiP_init = 1 - phiR_init
cAA_init = growth.model.sstRNA_balance(nu_init, phiP_init, gamma_max, phiR_init, Kd)
gamma_init = growth.model.translation_rate(gamma_max, cAA_init, Kd)
Mr_init = phiR_init * M0
Mp_init = phiP_init * M0
cN_init = 100

T_START = 0 
T_END =  3  
N_STEPS = 1000
dt = (T_END - T_START) / N_STEPS
time_range = np.linspace(T_START, T_END, N_STEPS)
phiR_range = np.linspace(0, 1, 500)

dfs = []
for i, strat in enumerate(tqdm.tqdm(['constant', 'optimal', 'elongation'])): 
    out = np.zeros((5, len(time_range)))
    out[:, 0] = [M0, Mr_init, Mp_init, cAA_init, cN_init]
    phiR_val = np.zeros_like(time_range)
    phiR_val[0] = phiR_init
    for j in range(1, len(time_range)):
        params = out[:, j-1]
        if time_range[j] < time_shift:
            args = (gamma_max, nu_init, omega, phiR_init, phiP_init, Kd)
            _out = np.array(growth.model.batch_culture_self_replicator(
                            params,
                            0,
                            *args))
            out[:, j] = out[:, j-1] + (_out * dt)
            phiR_val[j] = phiR_init
        else:
            # Choose the strategy 
            if strat == 'constant':
                args = (gamma_max, nu_shift, omega, phiR_init, phiP_init, Kd)
                _out = np.array(growth.model.batch_culture_self_replicator(
                            params,
                            0,
                            *args))
                out[:, j] = out[:, j-1] + (_out * dt)
                phiR_val[j] = phiR_init
    
            else:  
                biomass_diff = np.zeros_like(phiR_range)
                cAA = np.zeros_like(phiR_range)
                for k, phi in enumerate(phiR_range):
                    args = (gamma_max, nu_shift, omega, phi, 1 - phi, Kd)
                    _out = np.array(growth.model.batch_culture_self_replicator(
                            params, 0, *args))
                    biomass_diff[k] = _out[0]
                    cAA[k] = params[-2] + (_out[-2] * dt)
                gamma = gamma_max * (cAA / (cAA + Kd))
                if strat == 'optimal':
                    ind = np.argmax(biomass_diff)
                    phi = phiR_range[ind]
                    args= (gamma_max, nu_shift, omega, phi, 1- phi, Kd)
                    _out = np.array(growth.model.batch_culture_self_replicator(
                                params, 0, *args))
                    out[:, j] = out[:, j-1] + (_out * dt)
                    phiR_val[j] = phi
                elif strat == 'elongation':
                    ind = np.argmax(gamma)
                    phi = phiR_range[ind]
                    args= (gamma_max, nu_shift, omega, phi, 1- phi, Kd)
                    _out = np.array(growth.model.batch_culture_self_replicator(
                                params, 0, *args))
                    phiR_val[j] = phi
                    out[:, j] = out[:, j-1] + (_out * dt)

    df = pd.DataFrame(out.T, columns=['M', 'Mr', 'Mp', 'cAA', 'cN'])
    df['phiR'] = phiR_val
    df['rel_biomass'] = df['M'] / M0
    df['strategy'] = strat
    df['time'] = time_range
    dfs.append(df)
df = pd.concat(dfs, sort=False)

#%%
alt.Chart(df).mark_line().encode(
        x=alt.X('time:Q'),
        y=alt.Y('phiR:Q'), #, scale=alt.Scale(type='log')),
        color=alt.Color('strategy:N'),
        strokeDash=alt.StrokeDash('strategy:N')
).interactive()
# %%
