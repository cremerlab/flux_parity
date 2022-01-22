#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import growth.model 
import growth.integrate
import growth.viz
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()

# Define constants
phiO_postshift = 0.55
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_shift = 2 
nu_init = np.linspace(0.005 * nu_shift, 0.99 * nu_shift, 10)
phiO_preshift = np.linspace(0.9, 0.55, len(nu_init))
total_time = 15
shift_time = 1 
dt = 0.0001
shift_magnitudes = pd.read_csv('../../data/main_figure_data/Fig5C_shift_magnitudes.csv')
#%%
# Iterate through shifts 
shift_df = []
shift_rate = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_init)):
 
    # Set up the init_args
    init_args = {'gamma_max':gamma_max,
                 'nu_max': nu,
                 'tau': tau,
                 'Kd_TAA': Kd_TAA,
                 'Kd_TAA_star': Kd_TAA_star,
                 'kappa_max': kappa_max,
                 'phi_O': phiO_preshift[i]}

    # Set up shift args
    shift_args = {'gamma_max':gamma_max,
                  'nu_max': nu_shift,
                  'tau': tau,
                  'Kd_TAA': Kd_TAA,
                  'Kd_TAA_star': Kd_TAA_star,
                  'kappa_max': kappa_max,
                  'phi_O': phiO_postshift}
    args = [init_args, shift_args]
    # Perform the shift
    shift = growth.integrate.nutrient_shift_FPM(args,
                                              shift_time, 
                                              total_time,
                                              dt=dt)
    lam_sat = growth.integrate.equilibrate_FPM(shift_args, t_return=2, dt=0.001)
    lam_sat = np.log(lam_sat[1][0] / lam_sat[0][0]) / 0.001

    # Assign the shift magnitude and compute some properties
    shift['nu_shift'] = nu
    shift['ratio'] = shift['TAA_star'].values / shift['TAA'].values
    shift['phi_Rb'] =  shift['ratio'].values / (shift['ratio'].values + tau)

    # compute the growth rate
    gr = list(np.log(shift['M'].values[1:]/shift['M'].values[:-1]) / dt)
    gr.append(gr[-1])
    shift['inst_growth_rate'] = gr
    shift_df.append(shift)

    # Compute the shift statistics
    lam_0 = gr[0]

    # Find the switch
    postshift_df = shift[(shift['shifted_time'] >= 0) & (shift['shifted_time'] <= (1/3))] # looking at the first 10 min.

    lam_1 = postshift_df['inst_growth_rate'].values.mean()

    shift_rate = shift_rate.append({'lam_0':lam_0,
                                    'lam_sat': lam_sat,
                                    'lam_1': lam_1,
                                    'nu_init': nu,
                                    'nu_shift': nu_shift}, 
                                    ignore_index=True)
    
    
shift_df = pd.concat(shift_df)

# %%
# compute the geom mean
shift_rate['lam1_lam0'] = shift_rate['lam_1'].values / shift_rate['lam_0'].values
shift_rate['lam0_lamsat'] = shift_rate['lam_0'].values  / shift_rate['lam_sat'].values
# %%
fig, ax = plt.subplots(1, 1, figsize=(2.4, 1.9))
cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=len(shift_magnitudes['source'].unique()))
counter = 0
for g, d in shift_magnitudes.groupby(['source']):
        ax.plot(d['total_shift_magnitude'], d['initial_shift_magnitude'], 
                marker=mapper[g]['m'], linestyle='none', ms=4.5, alpha=0.75,
                color=cmap[counter], markeredgecolor='k', markeredgewidth=0.5,
                label=g)
        counter += 1

ax.plot(shift_rate['lam0_lamsat'], shift_rate['lam1_lam0'], '--', color=colors['primary_black'], lw=1, zorder=1000)
ax.set_ylim([0.5, 4])
ax.set_xlim([0, 1.05])
ax.set_xlabel('total shift magnitude\n$\lambda_i^{(preshift)} / \lambda_i^{(postshift)}$')
ax.set_ylabel(r'$\lambda_i^{\dagger}  / \lambda_i^{(preshift)}$' + '\ninitial shift magnitude')
ax.legend()
plt.tight_layout()
plt.savefig('../../Figures/Fig5C2_spare_capacity.pdf', bbox_inches='tight')
# %%

# %%
