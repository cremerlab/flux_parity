#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import growth.integrate
import tqdm
import growth.viz
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()
# Load the erickson data
sectors = pd.read_csv('../../data/Erickson2017_sector_dynamics.csv')
mass_fraction = pd.read_csv('../../data/Erickson2017_shift_mass_fraction.csv')
od_data = pd.read_csv('../../data/Erickson2017_Fig1_shifts.csv')
gr_data = pd.read_csv('../../data/Erickson2017_Fig1_instant_growth_rates.csv')
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
gamma_max = const['gamma_max']
phiO = 0.55
Kd_cpc = const['Kd_cpc']
Kd_TAA = 3E-5
Kd_TAA_star = 3E-5
tau = 1
kappa_max = const['kappa_max']
lam = [[0.45, 0.85], [0.91, 0.45]]
phiRb = [[0.089, 0.140], [0.124, 0.089]]
phiO_shift = [[phiO + delta_phiO_upshift, phiO], [phiO, phiO+ delta_phiO_downshift]]
shifts = []

#%%
for i, shift_type in enumerate(['upshift']):#, 'downshift']):
    preshift_nu = growth.integrate.estimate_nu_FPM(phiRb[i][0], lam[i][0], const, phiO_shift[i][0])
    postshift_nu = growth.integrate.estimate_nu_FPM(phiRb[i][1], lam[i][1], const, phiO_shift[i][1])
 
    # peform the shift
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
    shift['type'] = shift_type
    gr = list(np.log(shift['M'].values[1:] / shift['M'].values[:-1]) /\
             (shift['time'].values[1:] - shift['time'].values[:-1]))
    gr.append(gr[-1])
    shift['instant_growth_rate'] = gr
    shifts.append(shift)

_shift_df = pd.concat(shifts, sort=False)
_shift_df['MRb_M'] = _shift_df['M_Rb'].values / _shift_df['M'].values


#%%
fig, ax = plt.subplots(1, 2, figsize=(4.5, 1.75))

ax[0].set_yscale('log', base=2)
shift_df = _shift_df[_shift_df['type']=='upshift']
ax[0].plot(shift_df['shifted_time'], 0.025 * shift_df['M'], '--', lw=1, 
            color=colors['dark_black'], zorder=1000)
ax[1].plot(shift_df['shifted_time'], shift_df['instant_growth_rate'], '--', lw=1, 
                    color=colors['dark_black'], zorder=1000)

for g, d in od_data.groupby(['type']):
    if g == 'upshift':
        ax[0].plot(d['time_hr'], d['od_600nm'], '.', ms=8,
                color=colors['dark_red'],
                markeredgecolor='k', markeredgewidth=0.5,
                alpha=0.5, label='Erickson et al., 2017')
for g, d in gr_data.groupby(['type']):
    if g == 'upshift':   
        ax[1].plot(d['time_hr'], d['instant_growth_rate_hr'], '.', ms=8,
                   color=colors['dark_red'],
                   markeredgecolor='k', markeredgewidth=0.5,
                   alpha=0.5)

# ax[0].vlines(0, 0.02, 1.28, 'k', lw=2, alpha=0.25)
# ax[1].vlines(0, 0.2, 1, 'k', lw=2, alpha=0.25)
ax[0].set_ylim([0.02, 1.28])
ax[1].set_ylim([0.2, 1.0])
ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax[0].set_yticks([0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
ax[0].set_xlim([-2.25, 2.5])
ax[0].set_yticklabels(['0.04', '0.08',  '0.16', '0.32', '0.64', '1.28'])
ax[0].set_ylabel('optical density [a.u.]')
ax[1].set_ylabel('$\lambda_{i}$ [hr$^{-1}$]\n instantaneous growth rate')
ax[0].set_xlabel('time from upshift [hr]')
ax[1].set_xlabel('time from upshift [hr]')
plt.tight_layout()
plt.savefig('../../figures/Fig5C1_nutritional_upshift.pdf', bbox_inches='tight')

# %%

# %%
