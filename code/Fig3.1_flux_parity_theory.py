#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import growth.model
import growth.integrate
import seaborn as sns
import tqdm
const = growth.model.load_constants()
source_colors = growth.viz.load_markercolors()
colors, palette = growth.viz.matplotlib_style()


# Load the parameter sweep data
sweep = pd.read_csv('../data/flux_parity_parameter_sweep.csv')
sweep = sweep[sweep['nu_max'] == 4.5]

# Load the expeirmental data
mass_fractions = pd.read_csv('../data/main_figure_data/ecoli_ribosomal_mass_fractions.csv')
elongation_rate = pd.read_csv('../data/main_figure_data/ecoli_peptide_elongation_rates.csv')

# %% Generate the heat maps
phiRb_map = np.zeros((len(sweep['tau'].unique()), len(sweep['kappa'].unique())))
i = 0
for g, d in sweep.groupby(['tau']): 
    phiRb_map[i, :] = d['phiRb'].values
    i += 1

# Compute the optimal growth rate and normalize the map
opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'], 
                                                  sweep['nu_max'].values[0], 
                                                  const['Kd_cpc'], 
                                                  const['phi_O'])
norm_phiRb = opt_phiRb / opt_phiRb

# Create the mesh for positioning of contours and labels
ind = np.arange(0, len(sweep['kappa'].unique()), 1)
X, Y = np.meshgrid(ind, ind)


#%%
# Evaluate the flux parity model over the metabolic rates
nu_range = np.linspace(0.05, 20, 300)
df = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    _args = {'gamma_max':const['gamma_max'], 
             'nu_max': nu,
             'Kd_TAA_star': const['Kd_TAA_star'],
             'Kd_TAA': const['Kd_TAA'],
             'phi_O': const['phi_O'],
             'kappa_max': const['kappa_max'],
             'tau': const['tau']}
    out = growth.integrate.equilibrate_FPM(_args, max_iter=100)
    gamma = const['gamma_max'] * out[-1] / (out[-1] + const['Kd_TAA_star'])
    balance = out[-1] / out[-2]
    phiRb = (1 - const['phi_O']) * balance / (balance + const['tau'])
    lam = gamma * phiRb
    _df = pd.DataFrame([np.array([phiRb, lam, gamma])], columns=['phiRb', 'growth_rate', 'gamma'])
    _df['nu_max'] = nu
    df = pd.concat([df, _df])



# %%
# Instantiate the figure and format the axes.
fig, ax = plt.subplots(1, 3, figsize=(7, 2.3))
ax[0].grid(False)
ax[0].set_xlabel('transcription rate\n$\kappa_{max}$ [hr$^{-1}$]')
ax[0].set_ylabel(r'$\tau$'+'\ncharging sensitivity')
ax[1].set_xlabel('growth rate\n$\lambda$ [hr$^{-1}$]')
ax[1].set_ylabel('$\phi_{Rb}$\n allocation towards ribosomes')
ax[2].set_xlabel('growth rate\n$\lambda$ [hr$^{-1}$]') 
ax[2].set_ylabel('$v_{tl}$ [AA / s]\ntranslation speed')
ax[0].set_xlim([0, 100])
ax[0].set_ylim([0, 100])
ax[1].set_ylim([0, 0.3])
ax[2].set_ylim([5.5, 20])

# # Add appropriate heatmap ticks and labels.
# inds = [[], []]
# labs = [[], []]
# for i, vals in enumerate(sweep['kappa'].unique(), sweep['tau'].unique()):
#     for j in range(np.log10(vals.min()).astype(int), np.log10(vals.max()).astype(int)+1):
#         _ind =  np.where(np.round(np.log10(vals), decimals=1) == j)
#         inds[i].append(_ind[0][0])
#         labs[i].append('10$^{%s}$' %int(j))
# ax[0].set_xticks(inds[0])
# ax[0].set_xticklabels(labs[0])
# ax[0].set_yticks(inds[1])
# ax[0].set_yticklabels(labs[1])

# Compute the optimal solution
nu_range = np.linspace(0.05, 20, 200)
opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'], nu_range,
                                        const['Kd_cpc'], const['phi_O'])
opt_lam = growth.model.steady_state_growth_rate(const['gamma_max'], opt_phiRb, 
                                        nu_range, const['Kd_cpc'], 
                                        const['phi_O'])
opt_gamma = growth.model.steady_state_gamma(const['gamma_max'], opt_phiRb, 
                                            nu_range, const['Kd_cpc'], 
                                            const['phi_O'])

# Plot the heatmap and contours
ax[0].imshow(norm_phiRb, cmap='mako', origin='lower')
cont_color = ['white', 'white', colors['primary_red'], 'white', 'white']
locations = [(20, 90), (30, 75), (50, 70), (40, 60), (30, 45)]
conts = ax[0].contour(X, Y, norm_phiRb, levels=[0.3, 0.7, 1, 1.5, 2], 
                      colors=cont_color)
ax[0].clabel(conts, conts.levels, inline=True, colors=cont_color, fontsize=6,
            manual=locations)

# Plot the data
for g, d in mass_fractions.groupby(['source']):    
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=4)
for g, d in elongation_rate.groupby(['source']):    
    ax[2].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], linestyle='none',
            marker=source_colors[g]['m'], markeredgecolor='k', alpha=0.75, 
            markerfacecolor=source_colors[g]['c'], markeredgewidth=0.25, ms=4)

# Plot the theory curves. 
ax[1].plot(opt_lam, opt_phiRb, '-', lw=1.5, color=colors['primary_blue'])
ax[1].plot(df['growth_rate'], df['phiRb'], '--', lw=1.5, color=colors['primary_red'])
ax[2].plot(opt_lam, opt_gamma * const['m_Rb'] / 3600, '-', lw=1.5, color=colors['primary_blue'])
ax[2].plot(df['growth_rate'], df['gamma'] *  const['m_Rb'] / 3600, '--', lw=1.5, color=colors['primary_red'])
plt.tight_layout()
plt.savefig('../figures/Fig3.1_flux_parity_plots.pdf')
# %%
