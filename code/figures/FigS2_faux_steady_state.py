#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import growth.integrate
import scipy.integrate
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()
data = pd.read_csv('../../data/main_figure_data/Fig4_ecoli_ribosomal_mass_fractions.csv')

gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']

#%%
# Estimate nu to achieve a target growth rate
fast_lam = 2.0
slow_lam = 0.2
fast_phiRb = 0.25
slow_phiRb = 0.03
fast_nu = growth.integrate.estimate_nu_FPM(fast_phiRb, fast_lam, const, phi_O)
slow_nu = growth.integrate.estimate_nu_FPM(slow_phiRb, slow_lam, const, phi_O)

# %%
# Set up integration parameters to show an example time scale
slow_lam = 0.25
max_gen = 10 
dt = 0.001
time_range = np.arange(0, max_gen/slow_lam, dt)

# Get the seed culture equilibrium
args = {'gamma_max': gamma_max,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'tau': tau,
        'kappa_max': kappa_max,
        'phi_O': phi_O,
        'nu_max': fast_nu} 
equil = growth.integrate.equilibrate_FPM(args)
equil_TAA_star = equil[-1]
equil_TAA = equil[-1]

equil_phiRb = equil[1] / equil[0]

# Get the preculture equilibrium
args['nu_max'] = slow_nu
slow_equil = growth.integrate.equilibrate_FPM(args)
slow_phiRb_equil = slow_equil[1] / slow_equil[0]

# Run the integration preculture
params = [1, equil_phiRb, 1 - phi_O - equil_phiRb, equil_TAA, equil_TAA_star]
args['nu_max'] = slow_nu
preculture = scipy.integrate.odeint(growth.model.self_replicator_FPM, params, 
                                    time_range, args=(args,))
preculture_df = pd.DataFrame(preculture, columns=['M', 'M_Rb', 'M_Mb', 'TAA', 'TAA_star'])
preculture_df['time'] = time_range
preculture_df['MRb_M'] = preculture_df['M_Rb'].values / preculture_df['M']
preculture_df['gens'] = time_range * slow_lam
_lam = list(np.log(preculture_df['M'].values[1:] / preculture_df['M'].values[:-1]) / dt)
_lam.append(_lam[-1])
preculture_df['phiRb_err'] = preculture_df['MRb_M'].values - slow_phiRb_equil
preculture_df['inst_lam'] = _lam
preculture_df['lam_err'] = preculture_df['inst_lam'].values - slow_lam
#%%
# Set up the figure canvas
fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))
ax = ax.ravel()
# Format axes
ax[0].set(ylabel='$M_{Rb} / M$\nribosome content',
          xlabel='time from inoculation [hr]')
ax[1].set(ylabel='difference from\n steady-state ribosome content',
          xlabel='time from inoculation [hr]',
          yscale='log')

gen_ticks = np.array([0, 2, 4, 6, 8, 10])
for a in ax:
    _ax = a.twiny()
    _ax.set_xticks(gen_ticks / slow_lam)
    _ax.set_xticklabels(gen_ticks)
    _ax.set_xlabel('time from inoculation [gen.]\n')
    _ax.grid(False)

ax[0].plot(preculture_df['time'], preculture_df['MRb_M'], lw=2)
ax[1].plot(preculture_df['time'], preculture_df['phiRb_err'], lw=2)
plt.tight_layout()
plt.savefig('../../figures/FigS2_preculture_dynamics_plots.pdf', bbox_inches='tight')

# %% 
# # Estimate nu for very slow growth
slow_lam = 0.02
slow_phi = 0.01
slow_nu = growth.integrate.estimate_nu_FPM(slow_phi, slow_lam, const, phi_O) 

#%%
# For each nu, compute the optimal growth rate
nu_range = np.linspace(slow_nu, fast_nu, 200)
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_range, 
                                                 const['Kd_cpc'], phi_O)
opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, nu_range, 
                                                const['Kd_cpc'], phi_O)
# Define properties of the actual integration
t_harvest = [8, 10, 12, 14, 16, 18, 20, 24, 28, 30, 48]
df = pd.DataFrame([])
for i, nu in enumerate(nu_range):
    time_range = np.arange(0, t_harvest[-1], dt)
    params = [1, equil_phiRb, 1 - equil_phiRb - phi_O, equil_TAA, equil_TAA_star]
    args['nu_max'] = nu
    out = scipy.integrate.odeint(growth.model.self_replicator_FPM,
                                params, time_range, args=(args,)) 
    ribo_content = out.T[:][1] / out.T[:][0]  
    for j, t in enumerate(t_harvest):
        ind = np.where(np.round(time_range, decimals=2) == t)[0][0]
        df = df.append({'nu_max': nu,
                        'lam': opt_lam[i],
                        'MRb_M': ribo_content[ind],
                        't_harvest': int(t)},
                        ignore_index=True)
    
# %%
fig, ax = plt.subplots(1, 1, figsize=(5, 3))
for g, d in data.groupby('source'):
    ax.plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
            marker=mapper[g]['m'], markerfacecolor=mapper[g]['c'],
            markeredgecolor='k', markeredgewidth=1, alpha=0.25, label='__nolegend__')
_colors = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=len(t_harvest)) 
count = 0 
for g, d in df.groupby(['t_harvest']): 
    ax.plot(d['lam'], d['MRb_M'], '--', lw=1, color=_colors[count], label=g)
    count += 1
leg = ax.legend(title='time from\ninoculation [hr]')
leg.get_title().set_fontsize(6)
ax.set_xlabel('growth rate\n$\lambda$ [hr$^{-1}$]')
ax.set_ylabel('$M_{Rb} / M$\nribosomal content')
plt.tight_layout()
plt.savefig('../../figures/FigS2_preculture_ribosome_content_data.pdf', bbox_inches='tight')
# %%
