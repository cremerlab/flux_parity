#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model
import tqdm
import scipy.integrate
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
def compute_nu(gamma_max, Kd, phiRb, lam, phi_O):
    """Estimates the metabolic rate given measured params"""
    return (lam / (1 - phiRb - phi_O)) * (((lam * Kd)/(gamma_max * phiRb - lam)) + 1)

def estimate_nu_ppGpp(phiRb, lam, phi_O, const, buffer=1, dt=0.0001, tol=2, verbose=False):
    nu = compute_nu(const['gamma_max'], const['Kd_cpc'], phiRb, lam, phi_O)
    lower = np.min([0.0001, np.abs(nu-buffer)])
    upper = nu + buffer
    nu_range = np.linspace(lower, upper, 100)
    converged = False
    ind = 0 
    diffs = []
    if verbose:
        iterator = enumerate(tqdm.tqdm(nu_range))
    else:
        iterator = enumerate(nu_range)
    for i, n in iterator:
       args = {'gamma_max': const['gamma_max'],
               'nu_max': n,
               'tau': const['tau'],
               'Kd_TAA': 3E-5,
               'Kd_TAA_star': 3E-5,
               'kappa_max':const['kappa_max'], 
               'phi_O':  phi_O}

       out = growth.model.equilibrate_ppGpp(args, dt=dt, t_return=2) 
       gr = np.log(out[1][0] / out[0][0]) / dt
       diff = np.round(gr - lam, decimals=tol)
       if diff == 0:
           print('Found satisfactory nu!')
           ind = i 
           converged = True
           break
       diffs.append(diff)
    if converged:
        return nu_range[ind]
    else:
        print('Metabolic rate not found over range. Try rerunning over a larger range.')
        return nu_range[np.argmin(diffs)]

# preshift_nu = estimate_nu_ppGpp(phiRb[0][0], lam[0][0], phiO_shift[0][0], const, verbose=True)
#%%
for i, shift_type in enumerate(['upshift']):#, 'downshift']):
    preshift_nu = estimate_nu_ppGpp(phiRb[i][0], lam[i][0], phiO_shift[i][0], const, verbose=True)
    postshift_nu = estimate_nu_ppGpp(phiRb[i][1], lam[i][1], phiO_shift[i][1], const, verbose=True)
 
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

    shift = growth.model.nutrient_shift_ppGpp([preshift_args, postshift_args],
                            total_time=6, shift_time=3)
    shift['type'] = shift_type
    gr = list(np.log(shift['M'].values[1:] / shift['M'].values[:-1]) /\
             (shift['time'].values[1:] - shift['time'].values[:-1]))
    gr.append(gr[-1])
    shift['instant_growth_rate'] = gr
    shifts.append(shift)

_shift_df = pd.concat(shifts, sort=False)

#    postshift_nu = 0.75 * compute_nu(gamma_max, Kd_cpc, phiRb[i][1], lam[i][1], phiO_shift[i][1])%%
_shift_df['MRb_M'] = _shift_df['M_Rb'].values / _shift_df['M'].values
# Compute the growth rate

#%%
fig, ax = plt.subplots(1, 2, figsize=(4, 2))

ax[0].set_yscale('log', base=2)
shift_df = _shift_df[_shift_df['type']=='upshift']
ax[0].plot(shift_df['shifted_time'], 0.025 * shift_df['M'], '--', lw=1, color=colors['primary_red'], zorder=1000)
ax[1].plot(shift_df['shifted_time'], shift_df['instant_growth_rate'], '--', lw=1, 
                    color=colors['primary_red'], zorder=1000)

for g, d in od_data.groupby(['type']):
    if g == 'upshift':
        ax[0].plot(d['time_hr'], d['od_600nm'], '.', ms=8,
                color=mapper['Erickson et al., 2017']['c'],
                markeredgecolor='k', markeredgewidth=0.5,
                alpha=0.5)
for g, d in gr_data.groupby(['type']):
    if g == 'upshift':   
        ax[1].plot(d['time_hr'], d['instant_growth_rate_hr'], '.', ms=8,
                   color=mapper['Erickson et al., 2017']['c'],
                   markeredgecolor='k', markeredgewidth=0.5,
                   alpha=0.5)

ax[0].vlines(0, 0.02, 1.28, 'k', lw=2, alpha=0.25)
ax[1].vlines(0, 0.2, 1, 'k', lw=2, alpha=0.25)
ax[0].set_ylim([0.02, 1.28])
ax[1].set_ylim([0.2, 1.0])
ax[0].set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax[1].set_xticks([-3, -2, -1, 0, 1, 2, 3])
ax[0].set_yticks([0.04, 0.08, 0.16, 0.32, 0.64, 1.28])
ax[0].set_yticklabels(['0.04', '0.08',  '0.16', '0.32', '0.64', '1.28'])
ax[0].set_ylabel('optical density [a.u.]')
ax[1].set_ylabel('$\lambda_{instant}$ [hr$^{-1}$]\n instantaneous growth rate')
ax[0].set_xlabel('time from upshift')
ax[1].set_xlabel('time from upshift')
plt.tight_layout()
plt.savefig('../../figures/Fig5X_nutritional_upshift.pdf', bbox_inches='tight')

# %%

# %%
