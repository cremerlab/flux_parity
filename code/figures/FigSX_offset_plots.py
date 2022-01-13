#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz 
import growth.model
import scipy.stats
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()

# Load the necessary constants
gamma_max = const['gamma_max']
lin_nu_max = np.linspace(2, 10, 200)
nu_max = np.linspace(0.01, 20, 300)

# Load the E. coli data 
data = pd.read_csv('../../data/ecoli_ribosomal_mass_fractions.csv')
data.head()
fast_data = data[data['growth_rate_hr'] >= 0.5]

# %%
# Do a simple linear regression on the data
data_popt = scipy.stats.linregress(fast_data['growth_rate_hr'], fast_data['mass_fraction'])
data_slope = data_popt[0]
data_intercept = data_popt[1]

#%% In the linear region of the model, estimate the slope 
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, lin_nu_max, const['Kd_cpc'], const['phi_O'])
opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, lin_nu_max, const['Kd_cpc'], const['phi_O'])
model_slope = np.mean(np.diff(opt_phiRb)/np.diff(opt_lam))
model_intercept = np.mean(opt_phiRb - opt_lam * model_slope)
 
# %%
# Set up the plot
fig, ax = plt.subplots(1, 2, figsize=(6, 3))
ax[0].axis('off')
ax[1].set_xlim([0, 2])
ax[1].set_ylim([0, 0.3])
ax[1].set_xlabel('growth rate\n' + '$\lambda$ [hr$^{-1}$]')
ax[1].set_ylabel('$\phi_{Rb}$' + '\nallocation towards ribosomes')
for g, d in fast_data.groupby(['source']):
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
            marker=mapper[g]['m'], markeredgecolor='k', markeredgewidth=0.5,
            markerfacecolor=mapper[g]['c'], label=g, alpha=0.5)

# Plot the two fits
lam_range = np.linspace(0, 2.5)
ax[1].plot(lam_range, data_slope * lam_range + data_intercept, '-', lw=2, 
           color=colors['primary_black'], label='linear fit to data\n' + r'$\lambda \geq$ 0.5 hr$^{-1}$')
ax[1].plot(lam_range, model_slope * lam_range + model_intercept, '--', lw=2,
        color=colors['primary_blue'], label='linear fit to optimal allocation model\n' + r'$\lambda \geq$ 0.5 hr$^{-1}$')
ax[1].legend(bbox_to_anchor=(-0.3, 1))
ax[0].plot([], [], '-', color=colors['primary_black'], lw=2, 
            label=f'slope = {data_popt[0]:0.2f} per hr\nintercept = {data_popt[1]:0.02f}') 
ax[0].plot([], [], '--', color=colors['primary_blue'], lw=2, 
        label=f'slope = {model_slope:0.3f} per hr\nintercept = {model_intercept:0.03f}') 
leg = ax[0].legend(bbox_to_anchor=(0.75, 0.12), title='estimated parameters')
leg.get_title().set_fontsize(6)
plt.savefig('../../figures/FigSX_offset_plots.pdf', bbox_inches='tight') 
# %%
