#%% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.model
import growth.viz 
colors, palette = growth.viz.matplotlib_style() 
const = growth.model.load_constants()

# Set the constants 
gamma_max = const['gamma_max']
nu_max = 4.5
Kd_cpc = const['Kd_cpc'] 
Kd_cnt = const['Kd_cnt']
phi_O = const['phi_O']
Y = const['Y']
c_nt = 100
phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
cpc = growth.model.steady_state_precursors(gamma_max, phiRb, nu_max, Kd_cpc, phi_O)

# set the time range
dt = 0.001
time_range = np.arange(0, 4 + dt, dt)
perturb = [0.5, 2]

# Compute the pre-perturbation
params = [1, phiRb, 1 - phi_O - phiRb, cpc, c_nt]
args = (gamma_max, nu_max, Y, phiRb, 1 - phi_O - phiRb, Kd_cpc, Kd_cnt)
out = scipy.integrate.odeint(growth.model.self_replicator, params, time_range, args)
pre_df = pd.DataFrame(out, columns=['M', 'M_Rb', 'M_Mb', 'c_pc', 'c_nt'])
pre_df['perturbation'] = perturb[0]
pre_df['dilution'] = True
pre_df['time'] = time_range


# Make the perturbation 
params = list(out[-1])
# params[3] *= perturb
dfs = []
for i, p in enumerate(perturb):
    params[3] = p * cpc

    # Integration with dilution
    args = (gamma_max, nu_max, Y, phiRb, 1 - phi_O - phiRb, Kd_cpc, Kd_cnt, False)
    out_dil = scipy.integrate.odeint(growth.model.self_replicator, params, time_range, args)
    dil_df = pd.DataFrame(out_dil, columns=['M', 'M_Rb', 'M_Mb', 'c_pc', 'c_nt'])
    dil_df['dilution'] = True
    dil_df['time'] = time_range + time_range.max() - dt
    dil_df['perturb'] = p

    # Integration without dilution
    args = (gamma_max, nu_max, Y, phiRb, 1 - phi_O - phiRb, Kd_cpc, Kd_cnt, True)
    out_nodil = scipy.integrate.odeint(growth.model.self_replicator, params, time_range, args)
    nodil_df = pd.DataFrame(out_nodil, columns=['M', 'M_Rb', 'M_Mb', 'c_pc', 'c_nt'])
    nodil_df['dilution'] = False
    nodil_df['time'] = time_range + time_range.max() - dt
    nodil_df['perturb'] = p
    dfs.append(dil_df)
    dfs.append(nodil_df)
df = pd.concat(dfs, sort=False)
# %%
fig, ax = plt.subplots(2, 1, figsize=(4, 3), sharex=True)
for g, d in df.groupby(['perturb', 'dilution']):
    if g[1] == True:
        ls = '-'    
        label = 'including dilution'
    else: 
        ls = '--'
        label = 'neglecting dilution'
    if g[0] < 1:
        _ax = ax[1]   
    else:
        _ax = ax[0]
    _ax.plot(d['time'], d['c_pc'] / Kd_cpc, ls, color=colors['primary_blue'], lw=1,
             label=label)
ax[0].legend()
for i, a in enumerate(ax):
    _time = pre_df['time'].values
    _cpc = pre_df['c_pc'].values
    _cpc[-1] = perturb[-(i + 1)] * cpc 
    a.plot(_time, _cpc / Kd_cpc, '-', color=colors['primary_blue'], lw=1)
    a.set(ylabel=r'$c_{pc}/K_M^{c_{pc}}$') 
    a.set_ylim([1, 9])
    a.vlines(time_range[-1], 1, 9, lw=2, color=colors['light_black'], alpha=0.45)
ax[1].set_xlabel('time [hr]')
ax[0].set_title(r'instantaneous influx of precursors $c_{pc} = 2\times c_{pc}^*$')
ax[1].set_title(r'instantaneous efflux of precursors $c_{pc} = 0.5\times c_{pc}^*$')
ax[1].legend()
plt.tight_layout()
fig.text(0.01, 0.92, '(A)', fontsize=8, fontweight='bold')
fig.text(0.01, 0.48, '(B)', fontsize=8, fontweight='bold')
plt.savefig('../figures/supplement_text/FigS1_dilution_effect.pdf', bbox_inches='tight')
# %%
