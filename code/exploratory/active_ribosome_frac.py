#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.viz
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

gamma_max = const['gamma_max']
nu_max = np.linspace(0.001, 5, 200)
Kd_cpc = const['Kd_cpc']
phi_O = 0.25

opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O)

tot_prot = 1E9
tot_ribo = opt_phiRb * tot_prot
n_ribo = tot_ribo / const['m_Rb']
rho = 3 / 500
tot_nt = rho * tot_prot
spacing = 200
n_sites = tot_nt / spacing
excess_ribos = n_ribo - n_sites
excess_ribos *= excess_ribos >= 0
f_a = (n_ribo - excess_ribos) / n_ribo

# compute the spacing
act_spacing = tot_nt / n_ribo

# %%
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].plot(lam, f_a, 'k-',  lw=2)
ax[1].plot(lam, act_spacing, 'k-', lw=2)
ax[1].set_yscale('log')
ax[1].hlines(200, 0, ax[1].get_xlim()[1], 'r', linestyle='--')
ax[0].set_ylim([0, 1])
# %%

# %%
n_ribo
# %%
n_ribo_inact
# %%
n_ribo
# %%
