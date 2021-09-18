#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
import growth.viz
import growth.model
import scipy.integrate
import tqdm
_ = growth.viz.matplotlib_style()

# Set the constants
gamma_max = 9.65 
nu_max = 2 
kappa = 2 
Kd = 0.025

# To convert OD units
OD_CONV = 1.5E17

# Set the initial conditions
M_init = 0.01 * OD_CONV
phiR_init = growth.model.phi_R_optimal_allocation(gamma_max, nu_max, Kd, 0) 
init_phiR = phiR_init
Mr_init = phiR_init * M_init
Mp_init = (1 - init_phiR) * M_init
lam = growth.model.steady_state_growth_rate(gamma_max, nu_max, phiR_init, 1- phiR_init, Kd)
cAA_init = growth.model.steady_state_tRNA_balance(nu_max, 1-phiR_init, lam)
# Set the integration function heavily reduced

def integrate_step(t, vars, phiR, gamma_max):
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


# Set a range of phiR 
phiR_range = np.linspace(0, 1, 1000)

# Set a time range to integrate
T_END = 6 
dt = 1 / 500 
time_range = np.arange(dt, T_END, dt)

# Set the output vector

biomass_phiR = np.zeros((len(phiR_range), len(time_range)))
phiR_vals = np.zeros_like(time_range)

sweep_dfs = []
for i, phi in enumerate(tqdm.tqdm(phiR_range)):
    params = [Mr_init, Mp_init, 0, cAA_init]
    args = (phi, gamma_max)
    out = scipy.integrate.odeint(integrate_step, params, time_range,
                                args=args, tfirst=True, rtol=1E-7, atol=1E-7)
    
    total_biomass = out[:, 0] + out[:, 1] + out[:, 2]
    diff_mu = np.log(total_biomass / (Mr_init + Mp_init)) / time_range
    realized_phiR = out[:, 0] / total_biomass
    
    # Look at the sweep
    _df = pd.DataFrame([])
    _df['biomass'] = total_biomass
    _df['instant_gr'] = diff_mu 
    _df['set_phiR'] = phi
    _df['real_phiR'] = realized_phiR
    _df['integrated_time'] = time_range
    sweep_dfs.append(_df)
time_sweep_df = pd.concat(sweep_dfs, sort=False)

#%%
dfs =[]
opt_df = pd.DataFrame([])
for g, d in time_sweep_df.groupby(['integrated_time']):
    d['instant_gr_norm'] = d['instant_gr'].values / d['instant_gr'].max()
    dfs.append(d)
    ind = np.argmax(d['instant_gr'].values)
    opt_df = opt_df.append({'integrated_time': g, 
                            'opt_phiR': d['set_phiR'].values[ind],
                            'real_phiR':d['real_phiR'].values[ind]},
                            ignore_index=True)
    
phi_sweep_df = pd.concat(dfs, sort=True)

# %%
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].set_xlabel('$\phi_R$')
ax[0].set_ylabel('normalized instantaneous growth rate')
ax[1].set_xlabel('integrated time [hr]')
ax[1].set_ylabel('$\phi_R$ with $\mu^{(max)}$')

# ax[1].set_ylim([0.1, 1.1])
# ax[1].set_xlim([8, 8.5])
cmap = sns.color_palette('crest', n_colors=len(time_range)).as_hex()
i = 0
for g, d in phi_sweep_df.groupby('integrated_time'):
    if i%100 == 0:
        ax[0].plot(d['set_phiR'], d['instant_gr_norm'], '-', lw=1, color=cmap[i])
    i += 1


ax[1].scatter(opt_df['integrated_time'], opt_df['opt_phiR'], 
            s=2, label='instantaneous optimal $\phi_R$ (exponential)',
            c=cmap)
# ax[1].plot(opt_df['integrated_time'], opt_df['real_phiR'], '-o', ms=2, label='realized $\phi_R$')
ax[1].hlines(phiR_init, 0, opt_df['integrated_time'].max(), linestyle=':', label='steady state $\phi_R$')
ax[1].legend()
plt.tight_layout()
plt.savefig(f'./instantaneous_parameter_sweep_kappa{kappa}_phiRinit{init_phiR:0.2f}.pdf', bbox_inches='tight')
# %%

# %%
