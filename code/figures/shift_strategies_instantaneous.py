#%%
import numpy as np 
import pandas as pd 
import altair as alt 
import scipy.integrate
import growth.viz 
import growth.model
colors, palette = growth.viz.altair_style()

# Set the initial conditions
Kd = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23)
OD_CONV = 1.5E17
VOL = 1E-3
M0 = 0.01 * OD_CONV
nu_init = 3
nu_shift = 5 
gamma_max = 20 * 3600 / 7459
time_shift = 2

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
cN_init = 0.01 

# Define the strategies post shift
phi_R_const = phiR_init
phi_R_opt = growth.model.phi_R_optimal_allocation(gamma_max, nu_shift, Kd, phi_O=0)
phi_R_trans = growth.model.phi_R_specific_translation(gamma_init, gamma_max, nu_shift, Kd)


# Define the time range
preshift_time = np.linspace(0, 1, 400)
postshift_time = np.linspace(0, 3, 600)
time_range = np.linspace(0, 5, 800)
strategies = ['optimal', 'constant', 'translation']
dfs = []

# Perform the integration for the preshift 
preshift_params = [M0, Mr_init, Mp_init, cAA_init, cN_init]
preshift_args = (gamma_max, nu_init, omega, phiR_init, phiP_init, Kd)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator,
                                         preshift_params, preshift_time, 
                                         args=preshift_args)

for i, phi in enumerate([phi_R_opt, phi_R_const, phi_R_trans]):
    out = np.zeros((5, len(preshift_time) + len(postshift_time)))
    out[:, :len(preshift_time)]  = preshift_out.T                                   

    # Perform the integration for the postshift 
    postshift_params = preshift_out.T[:, -1]
    postshift_args = (gamma_max, nu_shift, omega, phi, 1 - phi, Kd)
    postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator,
                                         postshift_params, postshift_time, 
                                         args=postshift_args)
    out[:, :len(postshift_time)] = postshift_out.T
    df = pd.DataFrame(out.T, columns=['biomass', 'ribosomal', 'metabolic', 'c_AA', 'c_N'])
    df['time'] = np.concatenate([preshift_time, postshift_time + preshift_time.max()]) 
    df['strategy'] = strategies[i]
    dfs.append(df)
df = pd.concat(dfs, sort=False)
df['rel_biomass'] = df['biomass'].values / M0
#%%

biomass_upshift = alt.Chart(df).mark_line().encode(
        x=alt.X('time:Q', title='time [hr]'),
        y=alt.Y('rel_biomass:Q', title='relative biomass'),
        color=alt.Color('strategy:N')
)
biomass_upshift
#%%
    # out[0, 0] = M0
    # out[1, 0] = Mr_init
    # out[2, 0] = Mp_init
    # out[3, 0] = cAA_init
    # out[4, 0] = cN_init
    # for j in range(1, len(time_range)): 
    #     params = out[:, j-1]
    #     if time_range[j] < time_shift:
    #         nu = nu_init
    #         phiR = phiR_init
    #         phiP = 1 - phiR_init
    #     else:
    #         nu = nu_shift
    #         phiR = phi
    #         phiP = 1 - phiR
    #     deriv = growth.model.batch_culture_self_replicator(params, 0, gamma_max, nu,
    #                                                         omega=omega, phi_R=phiR, 
    #                                                         phi_P=phiP,
    #                                                         Kd_cAA=Kd)
    #     out[:, j] = out[:, j-1] + deriv

    # # Assemble the dataframe
    # df = pd.DataFrame(out.T, columns=['biomass', 'ribosomal_mass', 'metabolic_mass', 'precursor_conc', 'nutrient_conc'])
    # df['strategy'] = strategies[i]
    # df['time'] = time_range
    

# %%
# 