#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.model
import growth.integrate
import growth.viz
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
# %%
nu_max = 4.5
gamma_max = const['gamma_max']
Kd_TAA_star = const['Kd_TAA_star']
Kd_TAA = const['Kd_TAA']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']

# Determine what the steady-state composition is
args = {'gamma_max':  gamma_max,
        'nu_max': nu_max,
        'Kd_TAA_star': Kd_TAA_star,
        'Kd_TAA': Kd_TAA,
        'tau': tau,
        'kappa_max': kappa_max,
        'phi_O': phi_O}
out = growth.integrate.equilibrate_FPM(args)
balance = out[-1]/out[-2]
ss_phiRb = (1 - phi_O) * balance / (balance + tau) 
ss_balance = balance
ss_MRbM = out[1]/out[0]

# Set starting conditions as different charging balances
M0 = 1E9
high_balance = 10
low_balance = 0.01

# Set the initial parameters
nu_max = np.array([1.5, 2.5, 3.5, 5.5, 6.5, 7.5])
relax_df = pd.DataFrame()
dt = 0.0001
time = np.arange(0, 2, dt)
for i, nu in enumerate(nu_max):
    init_args = {'gamma_max':gamma_max,
                 'nu_max': nu,
                 'Kd_TAA': Kd_TAA,
                 'Kd_TAA_star': Kd_TAA_star,
                 'kappa_max': kappa_max,
                 'tau': tau, 
                 'phi_O': phi_O}
    out = growth.integrate.equilibrate_FPM(init_args)
    init_params = [M0, out[1]/out[0] * M0, out[2]/out[0] * M0, out[-2], out[-1]]
    init_integration = scipy.integrate.odeint(growth.model.self_replicator_FPM,
                                        init_params, time, args=(args,))
    _df = pd.DataFrame(init_integration, columns=['M', 'M_Rb', 'M_Mb', 'TAA', 'TAA_star'])
    _df['time_hr'] = time
    _df['balance'] = _df['TAA_star'] / _df['TAA']
    _df['MRb_M'] = _df['M_Rb'] / _df['M']
    _df['phi_Rb'] = (1 - const['phi_O']) * _df['balance'] / (_df['balance'] + tau)
    inst_gr = list(np.log(_df['M'].values[1:]/_df['M'].values[:-1]) / dt)
    inst_gr.append(inst_gr[-1])
    _df['inst_gr'] = inst_gr
    _df['nu'] = nu
    relax_df = pd.concat([relax_df,_df], sort=False)


# %%

# Instantiate the canvas and format axes
fig, ax = plt.subplots(1, 3, figsize=(6,2))
ax[0].axis(False)
ax[1].set_xlim([0, 2])
ax[2].set_xlim([0, 2])
ax[2].set_ylim([0, 3])
ax[1].set_ylim([0, 3])
ax[1].hlines(1, 0, 2, linestyle='--', color=colors['primary_red'], lw=1.5, zorder=1000)
ax[2].hlines(1, 0, 2, linestyle='--', color=colors['primary_red'], lw=1.5, zorder=1000)
ax[1].set_xlabel('time [hr]', fontsize=6)
ax[2].set_xlabel('time [hr]', fontsize=6)
ax[1].set_ylabel('$\phi_{Rb}/\phi_{Rb}^*$\nrelative ribosomal allocation', fontsize=6)
ax[2].set_ylabel(r'$\frac{M_{Rb}}{M} / \frac{M_{Rb}^*}{M^*}$'+'\nrelative ribosome content', fontsize=6)
# ax[2].set_yticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
# ax[2].hlines(1, 0, 2, linestyle='--', color=colors['primary_red'], lw=1.5, zorder=1000)
i = 0
cmap = sns.color_palette('mako_r', n_colors=len(nu_max))
for g, d in relax_df.groupby(['nu']):
    ax[1].plot(d['time_hr'], d['phi_Rb']/ss_phiRb, '-', color=cmap[i], lw=1.5)
    ax[2].plot(d['time_hr'], d['MRb_M']/ss_phiRb, '-', color=cmap[i], lw=1.5)
    i+=1
plt.tight_layout()
plt.savefig('../figures/main_text/Fig4_theory.pdf', bbox_inches='tight')
# %%
