#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import growth.integrate
import tqdm
const = growth.model.load_constants()
colors, palette = growth.viz.matplotlib_style()
# %%
gamma_max = const['gamma_max']
Kd_TAA_range = [3E-6, 3E-5, 3E-4]
Kd_TAA_star_range = [3E-6, 3E-5, 3E-4]
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = const['phi_O']
nu_max = 4
phiRb_range = np.linspace(0.001, 1 - phi_O - 0.001, 100)
Kd_sweep_df = pd.DataFrame([])
for Kd, Kd_star in zip(Kd_TAA_range, Kd_TAA_star_range):
    for i, phiRb in enumerate(tqdm.tqdm(phiRb_range)):
        if Kd < Kd_star:
            label = r'$K_D^{tRNA} < K_D^{tRNA^*}$'
        elif Kd > Kd_star:
            label = r'$K_D^{tRNA} > K_D^{tRNA^*}$'
        else:
            label = r'$K_D^{tRNA} = K_D^{tRNA^*}$' 
        # Set the arguments
        args = {'gamma_max': gamma_max,
                'nu_max': nu_max,
                'Kd_TAA': Kd,
                'Kd_TAA_star': Kd,
                'tau': tau,
                'kappa_max':kappa_max,
                'phi_O':phi_O,
                'phiRb': phiRb}
        out = growth.integrate.equilibrate_FPM(args) 
        gamma = gamma_max * (out[-1] / (out[-1] + Kd_star))
        lam = gamma * phiRb
        Kd_sweep_df = Kd_sweep_df.append({'phiRb': phiRb,
                                          'Kd_TAA': Kd,
                                          'Kd_TAA_star': Kd_star,
                                          'gamma': gamma,
                                          'lam': lam,
                                          'label': label},
                                          ignore_index=True)
# %%
fig, ax = plt.subplots(1, 2, figsize=(4, 2))
for g, d in Kd_sweep_df.groupby(['Kd_TAA']):
    ax[0].plot(d['phiRb'], d['lam'], lw=1, label=g)
ax[0].legend()



# %%
