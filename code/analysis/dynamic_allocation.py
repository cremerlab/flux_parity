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

# Set the constants
gamma_max = 9.65
nu_max = 10
Kd = 0.025
# To convert OD units
OD_CONV = 1.5E17

# Set the initial conditions
M_init = 0.01 * OD_CONV
phiR_init = growth.model.phi_R_optimal_allocation(gamma_max, nu_max, Kd, 0) 
Mr_init = phiR_init * M_init
Mp_init = (1 - phiR_init) * M_init
cAA_init = growth.model.sstRNA_balance(nu_max, 1 - phiR_init, gamma_max, phiR_init, Kd)
mAA_init = cAA_init * M_init
# Set the integration function heavily reduced
def integrate_step(t, vars, phiR, gamma_max):
    Mr, Mp, mAA = vars

    # Compute cAA
    cAA = mAA / (Mr + Mp)

    # Compute the elongation rate
    gamma = gamma_max * (cAA / (cAA + Kd))

    # Biomass dynamics
    dM_dt = gamma * Mr 

    # Precursor dynamics
    dmAA_dt = nu_max * Mp - gamma * Mr

    # Allocation
    dMr_dt = phiR * dM_dt
    dMp_dt = (1 - phiR) * dM_dt
    return [dMr_dt, dMp_dt, dmAA_dt]


# Set a range of phiR 
phiR_range = np.linspace(0, 1, 500)

# Set a time range to integrate
T_END = 1
N_STEPS = 100
dt = T_END / N_STEPS
time_range = np.linspace(0, T_END, N_STEPS)

# Set the output vector
out = np.zeros((3, len(time_range)))
out[:, 0] = [Mr_init, Mp_init, mAA_init]

biomass_phiR = np.zeros((len(phiR_range), len(time_range)))
phiR_vals = np.zeros_like(time_range)
phiR_vals[0] = phiR_init
# Iterate through the time steps and integrate
for i in tqdm.tqdm(range(1, len(time_range))):
    params = out[: ,i - 1]
    outs = []
    for k, phi in enumerate(phiR_range):
        _out = scipy.integrate.solve_ivp(integrate_step, [0, dt], params, 
                                        method='Radau', t_eval=[dt], args=(phi,gamma_max))
        _out = _out['y'].T[0]
        biomass_phiR[k, i] = _out[0] + _out[1] 
        outs.append(_out)
    ind = np.argmax(biomass_phiR[:, i]) 
    phiR_vals[i] = phiR_range[ind]
    out[:, i] = outs[ind] 
# %%
df = pd.DataFrame(out.T, columns=['Mr', 'Mp', 'mAA'])
df['rel_biomass'] = (df['Mr'].values + df['Mp'].values) / M_init
df['phiR'] = phiR_vals
df['cAA'] = df['mAA'].values / (df['Mr'].values + df['Mp'].values)
df['time'] = time_range

base = alt.Chart(df)
M_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('rel_biomass:Q', title='relative biomass', 
                    scale=alt.Scale(type='log'))
            ).properties(width=250, height=250)
cAA_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('cAA:Q', title='precursor abundance'),
            ).properties(width=250, height=250)
phiR_plot = base.mark_line().encode(
            x=alt.X('time:Q', title='time [hr]'),
            y=alt.Y('phiR:Q', title='ribosome allocation ΦR'), 
            ).properties(width=250, height=250)


layout = (M_plot  | cAA_plot | phiR_plot)
# altair_saver.save(layout, './nutrient_shift_dynamic_reallocation.pdf')
layout

# %%
