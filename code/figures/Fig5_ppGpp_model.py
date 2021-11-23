#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import matplotlib.pyplot as plt 
import growth.model
import growth.viz
import tqdm
colors, palette = growth.viz.matplotlib_style()
mapper = growth.viz.load_markercolors()
const = growth.model.load_constants()

#%% Load the data
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[mass_frac['organism']=='Escherichia coli']
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong_rate = elong_rate[elong_rate['organism']=='Escherichia coli']
tRNA = pd.read_csv('../../data/tRNA_abundances.csv')
chlor = pd.read_csv('../../data/Scott2010_chloramphenicol_ribosome_content.csv')
lacZ = pd.read_csv('../../data/Scott2010_lacZ_overexpression.csv')

#%%
# Define the parameters
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
nu_max = np.linspace(0.1, 50, 30)
Kd_TAA = 3E-5 #const['Kd_TAA'] #1E-5 #in M, Kd of uncharged tRNA to  ligase
Kd_TAA_star = 3E-5 #const['Kd_TAA_star'] #1E-5
kappa_max = 0.0001 #const['kappa_max']
tau = 1 # const['tau']
phi_O = 0.55

# Compute the optimal scenario
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu_max, Kd_cpc, phi_O)
opt_mu = growth.model.steady_state_growth_rate(gamma_max,  opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cpc, phi_O) * 7459 / 3600


# #############################################################################  
# Numerically compute the optimal scenario
# #############################################################################  
dt = 0.0001
time_range = np.arange(0, 200, dt)
ss_df = pd.DataFrame([])
total_tRNA = 0.0004
T_AA = total_tRNA / 2
T_AA_star = total_tRNA / 2
for i, nu in enumerate(tqdm.tqdm(nu_max)):
    # Set the intitial state
    _opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu, Kd_cpc) 
    M0 = 1E9
    phi_Mb = 1 -  _opt_phiRb
    M_Rb = _opt_phiRb * M0
    M_Mb = phi_Mb * M0
    params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
    args = {'gamma_max': gamma_max,
            'nu_max':  nu,
            'tau': tau,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star':Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O}

    # Integrate
    _out = growth.model.equilibrate_ppGpp(args)

    # Compute the final props
    ratio = _out[-1] / _out[-2]
    tRNA_abund = _out[-2] + _out[-1]
    biomass = _out[0]
    gamma = gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star)

    ss_df = ss_df.append({'phi_Rb': _out[1] / _out[0],
                          'lam': gamma * (_out[1] /_out[0]),
                          'gamma': gamma * 7459 / 3600,
                          'balance': ratio,
                          'tot_tRNA_abundance': tRNA_abund,
                          'biomass': biomass,
                          'tRNA_per_ribosome': (tRNA_abund * biomass) / (_out[1]/7459), 
                          'nu_max': nu},
                          ignore_index=True)
#%%
# #############################################################################  
#  Numerically integrate the chlor scenario 
# #############################################################################  
# Identify the nu maxes to consider
def compute_nu(gamma_max, Kd, phiRb, lam):
    """Estimates the metabolic rate given measured params"""
    return (lam / (1 - phiRb - phi_O)) * (((lam * Kd)/(gamma_max * phiRb - lam)) + 1)

chlor_init = chlor[chlor['chlor_conc_uM']==0]
phiRb_targets = [i for i in chlor_init['mass_fraction'].values]
lam_targets = [i for i in chlor_init['growth_rate_hr'].values]

# Nudge nu by 10% to account for end product inhibition.
nu_targets = [1.25 * compute_nu(gamma_max, Kd_cpc, p, l) for p, l in zip(phiRb_targets, lam_targets)]

# Define a range of chlor concs to consider 
chlor_conc = np.linspace(0, 12, 100) * 1E-6

# Define constants for chlor binding
Kd_chlor = 5E-8

# Find the seed culture equilibrium
seed_phiRb = 0.21
seed_lam = 1.58
seed_phiMb = 1 - phi_O - seed_phiRb
nu_seed = compute_nu(gamma_max, Kd_cpc, seed_phiRb, seed_lam) # RDM + glucose condition from scott
init_params = [1, seed_phiRb, seed_phiMb, 1E-5, 1E-5]
args = {'gamma_max': gamma_max,
        'nu_max': nu_seed,
        'tau': tau,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max': kappa_max,
        'phi_O':  phi_O}
out = growth.model.equilibrate_ppGpp(args)

eq_phiRb = out[1]/out[0]
eq_phiMb = 1 - eq_phiRb - phi_O
eq_TAA = out[-2]
eq_TAA_star = out[-1]
M0 = 1E9
init_params = [M0, eq_phiRb * M0, (1 - phi_O - eq_phiRb) * M0, eq_TAA, eq_TAA_star]
dt = 0.001
time = np.arange(0, 18, dt) # Growing for 18 hours.
# Iterate through the nu targets
chlor_df = pd.DataFrame([])
for i, nu in enumerate(nu_targets):
    for j, c in enumerate(chlor_conc):
        args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O, c, Kd_chlor)
        out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp_chlor, init_params,
                            time, args=args)
        _out = out[-1]
        gr = np.log(out[-1][0] / out[-2][0]) / dt

        chlor_df = chlor_df.append({'MRb_M': _out[1]/_out[0],
                                    'lam':gr,
                                     'nu': nu,
                                     'conc': c}, ignore_index=True)

#%%
# #############################################################################  
#  Numerically integrate the overexpression scenario
# #############################################################################  
# Find the lacZ target nu
lacZ_init = lacZ[lacZ['phi_X']==0]
# phiRb_targets = [i for i in lacZ_init['mass_fraction'].values]
# lam_targets = [i for i in lacZ_init['growth_rate_hr'].values]
# nu_targets = [1.1 * compute_nu(gamma_max, Kd_cpc, p, l) for p,  l in zip(phiRb_targets, lam_targets)]
nu_targets = [9, 2.8, 1.8] # Determined empirically
phiX_range = np.linspace(0, 0.35, 10)
phiX_df = pd.DataFrame([])
dt = 0.0001
time = np.arange(0, 15, dt)
for i, nu in enumerate(tqdm.tqdm(nu_targets)):
    for j, x in enumerate(phiX_range):
        # Equilibrate to the steady state
        init_args = {'gamma_max':gamma_max,
                     'nu_max': nu,
                     'tau': tau,
                     'Kd_TAA': Kd_TAA,
                     'Kd_TAA_star': Kd_TAA_star,
                     'kappa_max':kappa_max,
                     'phi_O': phi_O + x}

        out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                            init_params, time, args=tuple(init_args.values()))
        # out = growth.model.equilibrate_ppGpp(init_args)

        # gamma = gamma_max * out[-1] / (out[-1] + Kd_TAA_star)
        # phiRb = out[1] / out[0]
        gr = np.log(out[-1][0] / out[-2][0]) / dt
        phiX_df = phiX_df.append({'phi_X':x,
                                  'nu': nu,
                                  'lam': gr}, 
                                  ignore_index=True)

#%%
fig, ax = plt.subplots(3, 2, figsize=(5.75,6.25))
ax[0, 0].axis('off')
ax[0, 1].set(ylim=[0, 0.45], xlim=[-0.05, 2.5], 
          xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
          ylabel='$\phi_{Rb}$\nallocation towards ribosomes')
ax[1, 0].set(ylim=[5, 20], xlim=[-0.05, 2.5], xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
         ylabel='$v_{tl}$ [AA / s]\ntranslation speed')

ax[1, 1].set(ylim=[0, 22], xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
         ylabel='tRNA per ribosome')
ax[2, 0].set(xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
             ylabel='$\phi_{Rb}$\nallocation towards ribosomes')
ax[2, 1].set(xlabel='allocation towards LacZ\n$\phi_{X}$',
             ylabel='$\lambda$ hr[$^{-1}$]\ngrowth rate')

sources = []
for g, d in mass_frac.groupby('source'):
    if g == 'Wu et al., 2021':
        continue
    sources.append(g)
    ax[0, 1].plot(d['growth_rate_hr'], d['mass_fraction'],  ms=4, marker=mapper[g]['m'],
            color=mapper[g]['c'], label=g, alpha=0.75,  markeredgecolor='k', 
            markeredgewidth=0.25, linestyle='none')

for g, d in elong_rate.groupby(['source']):
    if g == 'Wu et al., 2021':
        continue
    if g not in sources:
        sources.append(g)
    ax[1, 0].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'].values, marker=mapper[g]['m'],
                 ms=4, alpha=0.75, color=mapper[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
                 linestyle='none', label=g)

for g, d in tRNA.groupby(['source']):
    if g not in sources:
        sources.append(g)
    ax[1, 1].plot(d['growth_rate_hr'], d['tRNA_per_ribosome'], marker=mapper[g]['m'],
                ms=4, alpha=0.74, color=mapper[g]['c'], markeredgecolor='k', markeredgewidth=0.25,
                linestyle='none', label=g)

scott_marker = mapper['Scott et al., 2010']['m']
media = list(chlor['medium'].unique())
media.append('M63 + glucose')
_cmap = ['#632323', colors['dark_red'], colors['red'], '#AD4747', colors['primary_red'], '#E56363', colors['light_red']] 
_markers = ['o', 's', 'd', 'X', 'v', '^',  '>']
# sns.color_palette('rocket', n_colors=len(media))
cmap = {m:c for m, c in zip(media, _cmap)}
markers = {m:k for m, k in zip(media, _markers)}


for g, d in chlor.groupby(['medium']):
    ax[2, 0].plot(d['growth_rate_hr'], d['mass_fraction'], ms=4,
    marker=markers[g],  linestyle='none', markeredgecolor='k',
    alpha=0.75, color=cmap[g], label=g)

for g, d in lacZ.groupby(['medium']):
    ax[2, 1].plot(d['phi_X'], d['growth_rate_hr'], 'o', ms=4,
            marker=markers[g], linestyle='none', markeredgecolor='k',
            alpha=0.75, color=cmap[g], label=g)
for i, a in enumerate(ax.ravel()):
    if i >= 4:
        leg = a.legend(title='Scott et al., 2010')
        leg.get_title().set_fontsize(6)
    else:
        a.legend()
# Plot the theory curves
ax[0, 1].plot(opt_mu, opt_phiRb, '-', color=colors['primary_blue'], lw=1)
ax[0, 1].plot(ss_df['lam'], ss_df['phi_Rb'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[1, 0].plot(opt_mu, opt_gamma, '-', color=colors['primary_blue'], lw=1)
ax[1, 0].plot(ss_df['lam'], ss_df['gamma'], '--', color=colors['primary_red'], zorder=1000, lw=1)
ax[1, 1].plot(ss_df['lam'], ss_df['tRNA_per_ribosome'], '--', color=colors['primary_red'], zorder=1000,  lw=1)

counter = 0
for g, d in chlor_df.groupby('nu'):
    _d = d[d['lam'] >= 0.05]
    ax[2, 0].plot(d['lam'], d['MRb_M'], '--', lw=1, color=_cmap[counter]) 
    counter += 1
counter = 0
for g, d in phiX_df.groupby(['nu']):
    ax[2, 1].plot(d['phi_X'], d['lam'], '--', lw=1, color=_cmap[counter])
    counter += 1

plt.tight_layout()
# plt.savefig('../../figures/Fig5_ppGpp_model_plots.pdf')



# %%

# %%
