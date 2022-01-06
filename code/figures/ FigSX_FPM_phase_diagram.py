#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.integrate 
import growth.model 
import growth.viz 
import scipy.integrate
import tqdm
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
# %%
# Load the constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']
nu_max = 4.5
dt = 0.001

# Find the optimal solution 
args = {'gamma_max': gamma_max,
        'kappa_max': kappa_max,
        'nu_max': nu_max,
        'tau': tau,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'phi_O': phi_O}
out = growth.integrate.equilibrate_FPM(args)
# %%
opt_phiRb = out[1]/ out[0]
opt_TAA = out[-2]
opt_TAA_star = out[-1]
opt_ratio = opt_TAA_star / opt_TAA
opt_tot_tRNA = opt_TAA_star + opt_TAA
# %%
# Set the ranges of parameters to consider
time_range = np.arange(0, 10, dt)
M0 = 1E9
df = pd.DataFrame([])
ratio_range = np.linspace(0.1, 1, 100)
phiRb_range = np.linspace(0.001, 1 - phi_O - 0.001, 100)
time_range = np.arange(0, 20, dt)
for i, ratio in enumerate(tqdm.tqdm(ratio_range)):
    for j, phiRb in enumerate(phiRb_range):
        # Set up the starting parameters
        TAA_star = ratio * opt_tot_tRNA / (1 + ratio)
        params = [M0, phiRb * M0, (1 - phi_O - phiRb) * M0, opt_tot_tRNA - TAA_star, TAA_star]
        args = {'gamma_max':gamma_max,
                'nu_max': nu_max,
                'tau': tau,
                'kappa_max': kappa_max,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star': Kd_TAA_star,
                'phi_O': phi_O}
        out = scipy.integrate.odeint(growth.model.self_replicator_FPM,
                                    params, time_range, args=(args,))
        out = out[-1]
        df = df.append({'starting_ratio': ratio,
                        'starting_phiRb' : phiRb,
                        'ratio': out[-1] / out[-2],
                        'phiRb': (1 - phi_O) * ((out[-1] / out[-2]) / ((out[-1]/out[-2]) + tau))},
                        ignore_index=True)
        # Set up the data frame
        # df = pd.DataFrame(out, columns=['M', 'M_Rb', 'M_Mb', 'TAA', 'TAA_star'])
        # df['starting_ratio'] = ratio
        # df['starting_phiRb'] = phiRb
        # dfs.append(df)

# %%
# phase_df = pd.concat(dfs, sort=False)
# phase_df['ratio']  = phase_df['TAA_star'].values / phase_df['TAA'].values
# phase_df['phi_Rb'] = (1 - phi_O) * (phase_df['ratio'].values / (phase_df['ratio'].values + tau))

# %%
fig, ax = plt.subplots(1,1, figsize=(6, 4))
for g, d in df.groupby(['starting_ratio', 'starting_phiRb']):
    x, y = g
    dy = d['ratio'].values[0] - y
    dx = d['phiRb'].values[0] - x
    ax.arrow(x, y, dx, dy, head_width=0.01)

# ax.set_ylim([0, 1])
# ax.set_xlim([0, 0.45])
ax.set_ylim([0, 1])
ax.set_xlim([0, 1 - phi_O])

# %%
ratio_range = np.linspace(0.01, 0.8, 100)
phiRb_range = np.linspace(0.01, 0.3, 100)
RATIO, PHIR = np.meshgrid(ratio_range, phiRb_range)
RATIO_OUT, PHIR_OUT = np.zeros_like(RATIO), np.zeros_like(PHIR)
for i, ratio in enumerate(tqdm.tqdm(ratio_range)):
    for j, phiRb in enumerate(phiRb_range):
        # Set up the starting parameters
        TAA_star = ratio * opt_tot_tRNA / (1 + ratio)
        params = [M0, phiRb * M0, (1 - phi_O - phiRb) * M0, opt_tot_tRNA - TAA_star, TAA_star]
        args = {'gamma_max':gamma_max,
                'nu_max': nu_max,
                'tau': tau,
                'kappa_max': kappa_max,
                'Kd_TAA': Kd_TAA,
                'Kd_TAA_star': Kd_TAA_star,
                'phi_O': phi_O}
        out = growth.model.self_replicator_FPM(params, dt, args) 
            
        # Set up the data frame
        ratio_out = out[-1] / out[-2]
        phiR_out = (1 - phi_O) * ratio_out / (ratio_out + tau)
        RATIO_OUT[i, j] = ratio_out * dt
        PHIR_OUT[i, j] = phiR_out * dt




# %%
# plt.streamplot(phiRb_range, ratio_range, PHIR_OUT, RATIO_OUT) 
# plt.quiver(PHIR, RATIO, PHIR_OUT, RATIO_OUT, angles='xy')

# plt.ylim([0.550, 0.575])


# %%
PHIR

# %%
PHIR_OUT
# %%
np.shape(PHIR)
# %%
np.shape(PHIR_OUT)
# %%
