#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import growth.model 
import growth.viz
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Define constants
phi_O = 0.55
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_init = 0.05
nu_shift = np.linspace(2, 5, 100)
total_time =  4 
shift_time = 1 
dt = 0.001
# Iterate through shifts 
shift_df = []
shift_rate = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_shift)):
 
    # Set up the init_args
    init_args = {'gamma_max':gamma_max,
                 'nu_init': nu_init,
                 'tau': tau,
                 'Kd_TAA': Kd_TAA,
                 'Kd_TAA_star': Kd_TAA_star,
                 'kappa_max': kappa_max,
                 'phi_O': phi_O}

    # Set up shift args
    shift_args = {'gamma_max':gamma_max,
                  'nu_init': nu,
                  'tau': tau,
                  'Kd_TAA': Kd_TAA,
                  'Kd_TAA_star': Kd_TAA_star,
                  'kappa_max': kappa_max,
                  'phi_O': phi_O}
    args = [init_args, shift_args]

    # Perform the shift
    shift = growth.model.nutrient_shift_ppGpp(args,
                                              shift_time, 
                                              total_time)

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
    preshift_gr = gr[10]

    # Find the switch
    postshift_df = shift[shift['shifted_time'] > 0]
    ind = np.argmax(np.diff(postshift_df['inst_growth_rate'].values[1:]))
    postshift_gr = gr[ind + 1]
    shift_gr = shift[shift['shifted_time']==0]['inst_growth_rate'].values[0]

    shift_rate = shift_rate.append({'lam_0':preshift_gr,
                                    'lam_sat': postshift_gr,
                                    'lam_1': shift_gr,
                                    'nu_init': nu_init,
                                    'nu_shift': nu}, 
                                    ignore_index=True)
    
    
shift_df = pd.concat(shift_df)

# %%

fig, ax = plt.subplots(1, 1, figsize=(6,4))
cmap = sns.color_palette('mako', n_colors=len(nu_shift) + 2)
count = 0
for g, d in shift_df.groupby(['nu_shift']):
    ax.plot(d['shifted_time'], d['inst_growth_rate'], '-', lw=1, color=cmap[count])
    count += 1
# ax.set_xlim([-0.02, 0.025])    
# %%
# compute the geom mean
shift_rate['geom_mean'] = np.sqrt(shift_rate['lam_0'].values * shift_rate['lam_sat'].values)
shift_rate
# %%

fig, ax = plt.subplots(1, 1)
ax.plot(shift_rate['lam_0'].values / shift_rate['lam_sat'].values, shift_rate['lam_1'].values / shift_rate['lam_sat'].values, '-')


# %%
shift[shift['shifted_time']==0]
# %%
gr
# %%
shift['M']
# %%
shift
# %%
