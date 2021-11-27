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
phiO_preshift = 0.65
phiO_postshift = 0.55
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_init = np.linspace(0.5, 4.9, 50)
nu_shift =5 
total_time = 10
shift_time = 1 
dt = 0.001
data = pd.read_csv('../../data/Kohanim2018.csv')
#%%
# Iterate through shifts 
shift_df = []
shift_rate = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_init)):
 
    # Set up the init_args
    init_args = {'gamma_max':gamma_max,
                 'nu_init': nu,
                 'tau': tau,
                 'Kd_TAA': Kd_TAA,
                 'Kd_TAA_star': Kd_TAA_star,
                 'kappa_max': kappa_max,
                 'phi_O': phiO_preshift}

    # Set up shift args
    shift_args = {'gamma_max':gamma_max,
                  'nu_init': nu_shift,
                  'tau': tau,
                  'Kd_TAA': Kd_TAA,
                  'Kd_TAA_star': Kd_TAA_star,
                  'kappa_max': kappa_max,
                  'phi_O': phiO_postshift}
    args = [init_args, shift_args]

    # Perform the shift
    shift = growth.model.nutrient_shift_ppGpp(args,
                                              shift_time, 
                                              total_time)
    lam_sat = growth.model.equilibrate_ppGpp(shift_args, t_return=2, dt=0.001)
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
    preshift_gr = gr[10]
    # lam_sat = gr[-1]

    # Find the switch
    postshift_df = shift[shift['shifted_time'] > 0]
    ind = np.argmax(np.diff(postshift_df['inst_growth_rate'].values[1:]))
    postshift_gr = postshift_df['inst_growth_rate'].values[ind + 1]

    shift_rate = shift_rate.append({'lam_0':preshift_gr,
                                    'lam_sat': lam_sat,
                                    'lam_1': postshift_gr,
                                    'nu_init': nu,
                                    'nu_shift': nu_shift}, 
                                    ignore_index=True)
    
    
shift_df = pd.concat(shift_df)

# %%
fig, ax = plt.subplots(1, 1, figsize=(6,4))
cmap = sns.color_palette('mako', n_colors=len(nu_init) + 2)
count = 0
for g, d in shift_df.groupby(['nu_shift']):
    ax.plot(d['shifted_time'], d['inst_growth_rate'], '-', lw=1, color=cmap[count])
    count += 1
# ax.set_xlim([-0.02, 0.025])    
# %%
# compute the geom mean
shift_rate['geom_mean'] = np.sqrt(shift_rate['lam_0'].values * shift_rate['lam_sat'].values)
shift_rate['lam0_lamsat'] = shift_rate['lam_0'].values / shift_rate['lam_sat'].values
shift_rate['lam1_lam0'] = shift_rate['lam_1'].values  / shift_rate['lam_0'].values
# %%
plt.plot(shift_rate['lam0_lamsat'], shift_rate['lam1_lam0'], 'k--')
plt.plot(data['lam0_lamsat'], data['lam1_lam0'], 'o')
# %%
postshift_df
# %%
plt.plot(np.diff(postshift_df['inst_growth_rate'].values))
diff = np.diff(postshift_df['inst_growth_rate'].values)
ind = np.argmax(np.diff(postshift_df['inst_growth_rate'].values))
plt.plot(ind, diff[ind], 'o')
# %%
shift['M']
# %%
shift
# %%

# %%
