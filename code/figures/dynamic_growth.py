#%%
import numpy as np 
import pandas as pd 
import altair as alt 
from altair_saver import save
import scipy.integrate
import tqdm
import growth.viz 
import imp
import growth.model
alt.data_transformers.disable_max_rows()
colors, palette = growth.viz.altair_style(pub=True)
imp.reload(growth.model)

# Define the parameters
AVO = 6.022E23
VOL = 1E-3
OD_CONV = 1 / (7.7E-18)
gamma_max = 20 * 3600 / 7459 # Ribosomes per hr
Kd = 0.0013 * 20 
Km_0 = 1E5 # microgram glucose per liter

Km = (Km_0 * 1E-6) / (180.16)
phi_O = 0.35
phi_R = 0.4
phi_P = 1 - phi_R - phi_O
M0 = 0.1 * OD_CONV
M_P = phi_P * M0
M_R = phi_R * M0
c_AA = 0.001
c_N = .010
omega = 0.377 # * OD_CONV

def dynamics(params, t, gamma_max, nu_max, Kd, Km, omega, phi_R, phi_P):
    # Unpack parameters
    M, M_r, M_p, c_AA, c_N = params

    # Compute the number of precursors and the number of nutrients
    m_AA = c_AA * M
    m_N = c_N * 6.022E23 * 1E-3

    # Compute the capacities
    gamma = gamma_max * (c_AA / (c_AA + Kd))
    nu = nu_max * (c_N / (c_N + Km))

    # Biomass accumulation
    dM_dt = gamma * M_r

    # Resource allocation
    dMr_dt = phi_R * dM_dt
    dMp_dt = phi_P * dM_dt

    # Precursor dynamics
    dmAA_dt = nu * M_p - (1 + c_AA) * dM_dt
    dmN_dt = -nu * M_p / omega
    dcAA_dt = dmAA_dt / M
    dcN_dt = dmN_dt / (AVO * 1E-3)

    return [dM_dt, dMr_dt, dMp_dt, dcAA_dt, dcN_dt]

dfs = []
nu_range = np.linspace(0, 10, 25)
time_range = np.linspace(0, 5, 300)
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    params = [M0, M_R, M_P, c_AA, c_N] 
    args = (gamma_max, nu, Kd, Km, omega, phi_R, phi_P)
    out = scipy.integrate.odeint(dynamics, params, time_range, args=args)
    _df = pd.DataFrame(out, columns=['biomass', 'ribos', 'metabs', 'c_aa', 'c_n'])
    _df['c_n'] = _df['c_n'] / c_N
    _df['rel_biomass'] = _df['biomass'].values / M0
    _df['time'] = time_range * gamma_max
    _df['nu'] = nu
    dfs.append(_df)
df = pd.concat(dfs)
#%%
w, h = 150, 100 
lw = 1
scheme='magma'
biomass = alt.Chart(df, width=w, height=h).mark_line(size=lw).encode(
            x=alt.X('time:Q', title='time / γ_max'),
            y=alt.Y('rel_biomass:Q', title='relative biomass [M(t) / M0]',
                    scale=alt.Scale(type='log')),
            color=alt.Color('nu:Q', title='ν_max [T^-1]',
                            scale=alt.Scale(scheme=scheme, reverse=True))
)

caa = alt.Chart(df, width=w, height=h).mark_line(size=lw).encode(
          x=alt.X('time:Q', title='time x γ_max'),
          y=alt.Y('c_aa:Q', title='precursor concentration [m_AA / M]',
                  scale=alt.Scale(domain=[0, 0.04])),
          color=alt.Color('nu:Q', title='ν_max [T^-1]',
                        scale=alt.Scale(scheme=scheme, reverse=True)))

nuts = alt.Chart(df, width=w, height=h).mark_line(size=lw).encode(
        x=alt.X('time:Q', title='time x γ_max'),
        y=alt.Y('c_n:Q', title='relative nutrient concentration',
                axis=alt.Axis(format='%'), scale=alt.Scale(domain=[0, 1.05])),
        color=alt.Color('nu:Q', title='ν_max [T^-1]',
                   scale=alt.Scale(scheme=scheme, reverse=True))
)

layer = biomass & (caa | nuts)
save(biomass & (caa | nuts), '../../docs/figures/Fig2_integrated_dynamics_plots.pdf')
# %%
