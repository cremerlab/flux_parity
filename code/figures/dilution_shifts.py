#%%
import numpy as np 
import pandas as pd 
import altair as alt
import growth.viz
import growth.model
import scipy.integrate
import tqdm
colors, palette = growth.viz.altair_style()
alt.data_transformers.disable_max_rows()

# Set constants
Kd_cAA = (20 * 1E6 * 110) / (0.15E-12 * 6.022E23)
Kd_cN = 5E-4
cN = 0.01
VOL = 1E-3
OD_CONV = 1.5E17
gamma_max = 9.65
nu_max = np.array([5, 1, 1, 1])

opt_phiR = growth.model.phi_R_optimal_allocation(gamma_max, nu_max, 0, Kd_cAA)
const_phiR = opt_phiR[0]
cAA_opt = growth.model.sstRNA_balance(np.array(nu_max), 1 - opt_phiR, gamma_max, opt_phiR, Kd_cAA)
cAA_phiR = growth.model.phi_R_specific_cAA(cAA_opt[0], gamma_max, np.array(nu_max), Kd_cAA)

#%%
# Define the yield coefficient 
nutrient_mw = 180 * (6.022E23 / 110) # in AA mass units per mol 
omega = 0.3 * VOL * nutrient_mw # in units of Vol * AA mass / mol

# Set the integration variables
M0 = 0.01 * OD_CONV  
phiR = {'constant': np.array(len(nu_max) * [const_phiR]),
        'optimal': opt_phiR,
        'elongation': cAA_phiR}
cAA = 0.01
time_space = np.linspace(0, 2, 200)

dfs = []   
for k, v in phiR.items():
    Mr = v[0] * M0 
    Mp = M0 - Mr
    for i, nu in enumerate(nu_max):
        args = (gamma_max, nu, omega, v[i], 1 - v[i], Kd_cAA, Kd_cN)
        if i == 0:
            params  = [M0, Mr, Mp, cAA, cN]
            t = time_space
        else:  
            t = time_space + _df['time'].max()
            params = out[-1, :]
            params[-1] = cN
        out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator, params, time_space, args=args)
        _df = pd.DataFrame(out, columns=['M', 'Mr', 'Mp', 'cAA', 'cN'])
        _df['rel_biomass'] = _df['M'].values / M0
        _df['phiR_realized'] = _df['Mr'].values / _df['M']
        _df['phiR_prescribed'] = v[i] 
        _df['strategy'] = k
        _df['nu_max'] = nu
        _df['time'] = t
        dfs.append(_df)

dynamics = pd.concat(dfs, sort=False)



# %%
layout = alt.Chart(dynamics).mark_line().encode(x='time:Q', 
                                       y=alt.Y(
                                           'rel_biomass:Q', 
                                            scale=alt.Scale(type='log')
                                            ),
                                        color = alt.Color('strategy:N')
                                            )
# %%
