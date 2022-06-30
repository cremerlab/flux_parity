#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.integrate
import seaborn as sns
import tqdm
import growth.viz
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()
# Load the erickson data
sectors = pd.read_csv('../data/main_figure_data/Fig5C_Erickson2017_sector_dynamics.csv')
mass_fraction = pd.read_csv('../data/main_figure_data/Fig5C_Erickson2017_shift_mass_fraction.csv')

gr_data = pd.read_csv('../data/main_figure_data/Fig5C_Erickson2017_upshifts.csv')
gr_data = gr_data[(gr_data['preshift_medium']=='succinate') &
                  (gr_data['postshift_medium'].isin(['arabinose', 'glycerol', 'gluconate']))]
#%%
# Find the shifts in phiO
sectors = sectors[(sectors['sector']=='c_up') &
                  ((sectors['time_hr']==-0.25) | 
                  (sectors['time_hr']==4) |
                  (sectors['time_hr']==2.333))]
upshift = sectors[sectors['type']=='upshift']
downshift = sectors[sectors['type']=='downshift']
delta_phiO_upshift =  upshift['fraction'].values[0] - upshift['fraction'].values[1]
delta_phiO_downshift = downshift['fraction'].values[1] - downshift['fraction'].values[0]

#%%
# Load constants
phiO_preshift = 0.55 + delta_phiO_upshift
gamma_max = const['gamma_max']
phiO = const['phi_O']
Kd_cpc = const['Kd_cpc']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau'] 
kappa_max = const['kappa_max']
lam = [[0.45, 0.85], [0.45, 0.7], [0.5, 0.55]]
phiO_shift = [[phiO_preshift, 0.55],
              [phiO_preshift, 0.6],
              [phiO_preshift, 0.65]]

_postshift_nu = [2, 2.5, 3]
preshift_nu = growth.integrate.estimate_nu_FPM(0.05, 0.45, const, 0.65, 
                                               tol=3, nu_buffer=2, guess=1.45)

#%%
shifts = []
for i, shift_type in enumerate(['gluconate', 'glycerol', 'arabinose']):#, 'downshift']):
    postshift_nu = growth.integrate.estimate_nu_FPM(np.round(lam[i][1]/gamma_max, decimals=2), lam[i][1], const, phiO_shift[i][1],
                    tol=3, nu_buffer=2, guess=_postshift_nu[i], verbose=True)
    preshift_args = {'gamma_max':gamma_max,
        'nu_max': preshift_nu,
        'tau': tau,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max':kappa_max, 
        'phi_O': phiO_shift[i][0]}

    postshift_args = {'gamma_max':gamma_max,
        'nu_max': postshift_nu,
        'tau': tau,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max':kappa_max, 
        'phi_O': phiO_shift[i][1]}

    shift = growth.integrate.nutrient_shift_FPM([preshift_args, postshift_args],
                                                total_time=6, shift_time=3)
    gr = list(np.log(shift['M'].values[1:] / shift['M'].values[:-1]) /\
             (shift['time'].values[1:] - shift['time'].values[:-1]))
    gr.append(gr[-1])
    shift['type'] = shift_type
    shift['instant_growth_rate'] = gr
    shifts.append(shift)

_shift_df = pd.concat(shifts, sort=False)
_shift_df['MRb_M'] = _shift_df['M_Rb'].values / _shift_df['M'].values


#%%
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[1].axis(False)
ax[2].axis(False)
cmap = sns.color_palette(f"dark:{colors['primary_red']}", n_colors=3)
cmap = {'arabinose': cmap[0],
        'glycerol': cmap[1],
        'gluconate': cmap[2]}

for g, d in _shift_df.groupby(['type']):
    ax[0].plot(d['shifted_time'], d['instant_growth_rate'], '--', lw=1, 
                    zorder=1000,
                    color=cmap[g])

for g, d in gr_data.groupby(['postshift_medium']):
    ax[0].plot(d['shift_time_hr'], d['inst_growth_rate_hr'], 'o', ms=3.5, 
                   markeredgecolor='k', markeredgewidth=0.25, color=cmap[g],
                   alpha=0.75)

ax[0].set_ylim([0.2, 1.0])
ax[0].set_xlim([-2.5, 3])
ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax[0].set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax[0].set_ylabel('$\lambda_{i}$\n instantaneous growth rate [hr$^{-1}$]', fontsize=6)
ax[0].set_xlabel('time from upshift [hr]', fontsize=6)
plt.tight_layout()
plt.savefig('../figures/main_text/Fig4_nutritional_upshift.pdf', bbox_inches='tight')
# %%
