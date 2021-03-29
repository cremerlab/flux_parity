#%%
import numpy as np 
import pandas as pd 
import growth.viz
import growth.model
import altair as alt
import imp 
imp.reload(growth.model)
imp.reload(growth.viz)
colors, palette = growth.viz.altair_style()

# Define the translational parameters
gamma_max = (17.1 * 3600) / 7459 # per hr
nu_max = np.linspace(0.1, 10, 10) 
Kd = 2E-3

# Define allocation parameters
phi_O = 0.35
phi_R = np.linspace(0.001, 0.5, 100)
phi_P = 1 - phi_O - phi_R

# Compute the properties
lam_dfs = []
opt_df = pd.DataFrame({})
for i, nu in enumerate(nu_max):
    growth_rate = growth.model.growth_rate(nu, gamma_max, phi_R, phi_P, Kd)
    cAA = growth.model.tRNA_balance(nu, phi_P, growth_rate)
    gamma = growth.model.translation_rate(gamma_max, cAA, Kd)
    lam_df = pd.DataFrame([])
    lam_df['phi_R'] = phi_R
    lam_df['growth_rate'] = growth_rate
    lam_df['cAA'] = cAA
    lam_df['gamma'] = (gamma * 7459) / 3600
    lam_df['nu'] = nu  
    lam_dfs.append(lam_df)
    phi_R_opt = growth.model.optimal_phi_R(gamma_max, nu, Kd, phi_O)
    _phi_P = 1 - phi_O - phi_R_opt
    _opt_growth = growth.model.growth_rate(nu, gamma_max, phi_R_opt, _phi_P, Kd)
    _opt_cAA = growth.model.tRNA_balance(nu, _phi_P, _opt_growth) 
    _opt_gamma = growth.model.translation_rate(gamma_max, _opt_cAA, Kd)
    _opt_dict = {'nu': nu, 'growth_rate':_opt_growth, 'cAA':_opt_cAA,
                 'gamma':_opt_gamma * 7459 / 3600, 'phi_R':phi_R_opt}
    opt_df = opt_df.append(_opt_dict, ignore_index=True)
lam_df = pd.concat(lam_dfs, sort=False)

#%%
base = alt.Chart(lam_df).encode(
            x=alt.X('phi_R:Q', axis=alt.Axis(format='%'), title='allocation to translation'),
            color=alt.Color('nu:Q', title='nutritional capacity ν [hr^-1]'))
opt_base = alt.Chart(opt_df).encode(
            x=alt.X('phi_R:Q', axis=alt.Axis(format='%'), title='allocation to translation'),
            color=alt.Color('nu:N', title='nutritional capacity ν [hr^-1]'))

_growth = base.mark_line(width=200, height=200).encode(
                    y=alt.Y('growth_rate:Q', title='growth rate λ [hr^-1]'))
opt_growth = opt_base.mark_point(size=80, width=200, height=200).encode(
                    y=alt.Y('growth_rate:Q', title='growth rate λ [hr^-1]'))

_cAA = base.mark_line(width=200, height=200).encode(
                    y=alt.Y('cAA:Q', scale=alt.Scale(type='log'),
                            title='tRNA concentration'))
opt_cAA = opt_base.mark_point(size=80, width=200, height=200).encode(
                    y=alt.Y('cAA:Q', scale=alt.Scale(type='log'),
                            title='tRNA concentration'))

_gamma = base.mark_line(width=200, height=200).encode(
                    y=alt.Y('gamma:Q', title='elongation rate [AA/s]'))
opt_gamma = opt_base.mark_point(size=80, width=200, height=200).encode(
                    y=alt.Y('gamma:Q', title='elongation rate [AA/s]'))

(_growth + opt_growth) | (_cAA + opt_cAA) | (_gamma + opt_gamma)
# _growth  | _cAA | _gamma
#%%
fig, ax = plt.subplots(1, 3, figsize=(5, 1.5))
ax[0].plot(phi_R, growth_rate, '-')
ax[1].plot(phi_R, cAA, '-')
ax[1].set_yscale('log')
ax[2].plot(phi_R, gamma, '-')
# %%
