# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize
import seaborn as sns
import scipy.integrate
import growth.viz
import growth.model
import growth.size
colors, _ = growth.viz.matplotlib_style()

m_AA = 110 / 6.022E7  # Da to fg

elong = pd.read_csv('../../data/peptide_elongation_rates.csv')
elong = elong[elong['source'] == 'Dai et al., 2016']
elong.loc[elong['elongation_rate_aa_s'] == 15.0, 'growth_rate_hr'] = 0.69001


# %%
mass_frac = pd.read_csv('../../data/mass_fraction_compiled.csv')

dai = mass_frac[mass_frac['source'] == 'Dai et al., 2016']

# Assume no lag phase
lam_range = np.linspace(0.1, 2, 1000)

# Seed culture ribosomes
Nr_init = 1E5
t_preculture = 18  # in Hrs
ON_gen = 18 * lam_range
t_exp_gen = 3
tot_gen = ON_gen + t_exp_gen
tot_gen = 10
tot_gen = np.arange(4, 11)
frac = 0.5 ** tot_gen
residual_ribos = Nr_init * frac

mass_fg = growth.size.lambda2P(lam_range)
mass_AA = mass_fg / m_AA

palette = sns.color_palette('flare', n_colors=len(tot_gen))
fig, ax = plt.subplots(1, 1)
for i, n in enumerate(tot_gen):
    resids = (residual_ribos[i] * 7459)
    ax.plot(lam_range, resids, '-', label=n)  # , color=palette[i])
# plt.yscale('log')
ax.set_xlabel('true growth rate [hr]')
ax.set_ylabel('residual ribosome mass / total protein mass')
ax.legend(title='generations from seed')

# %%

# Assume there's at best the maximum according to Dai SI
tot_gen = 4
mass_fg = growth.size.lambda2P(dai['growth_rate_hr'].values)
Nr_init = 1E5
residual_frac = (Nr_init * 7459 * (0.5**tot_gen)) / (mass_fg/m_AA)
corrected = dai['mass_fraction'] - residual_frac
# corrected = dai['mass_fraction'] * (1 - 0.5**tot_gen)

# Define the organism specific constants
gamma_max_ecoli = 20 * 3600 / 7459
Kd_cAA_ecoli = 0.01
nu_max = np.linspace(0.001, 5, 300)

# Compute the theory curves

# Scenario III
opt_phiRb_ecoli = growth.model.phi_R_optimal_allocation(
    gamma_max_ecoli,  nu_max, Kd_cAA_ecoli)
opt_mu_ecoli = growth.model.steady_state_growth_rate(
    gamma_max_ecoli,  opt_phiRb_ecoli, nu_max, Kd_cAA_ecoli)
# opt_gamma_ecoli = growth.model.steady_state_gamma(gamma_max_ecoli, opt_phiRb_ecoli,  nu_max, Kd_cAA_ecoli) * 7459/3600

# %%

fig, ax = plt.subplots(1, 1)
ax.plot(dai['growth_rate_hr'], dai['mass_fraction'], 'o', label='uncorrected')
ax.plot(dai['growth_rate_hr'], corrected, 'o',
        markerfacecolor='none', markeredgecolor='k', label='corrected')
ax.plot(opt_mu_ecoli, opt_phiRb_ecoli, '-')
# %%

# %%
gamma_max = 9.65
M0 = 1E9
phi_Rb = 0.25
phi_Mb = 1 - phi_Rb
M_Rb = phi_Rb * M0
M_Mb = phi_Mb * M0
T_AA = 0.002
T_AA_star = 0.0002
tau = 3
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5
nu_preshift = 3.21
nu_postshift = [0.035, 0.11, 0.23, 0.349, 0.6, 0.879, 1.169] 
# nu_postshift = [
#                 0.035, 0.1413, 0.22,
#                 0.26, 0.326, 0.389,
#                 0.477, 0.49, 0.534,
#                 0.578, 0.664, 0.83, 
#                 0.85, 0.87, 0.894, 
#                 0.932, 1.187, 1.229,
#                 1.302, 1.276, 1.469,
#                 1.576, 1.755, 2.299,
#                 2.751]
                
dt = 0.0001
time_range = np.arange(0, 200, dt)
kappa_max = (88 * 5 * 3600) / 1E9

err_df = pd.DataFrame()
dfs = []
t_stop = 15
for i, nu in enumerate(nu_postshift):
    # Compute the preshift and postshift steady states
    init_params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
    preshift_args = (gamma_max, nu_preshift, tau, Kd_TAA_star,
                     Kd_TAA, 0, kappa_max, 0, False, True, True)
    postshift_args = (gamma_max, nu, tau, Kd_TAA_star, Kd_TAA,
                      0, kappa_max, 0, False, True, True)

    preshift_init = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                           init_params, time_range, args=preshift_args)
    postshift_init = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                            init_params, time_range, args=postshift_args)

    preshift_ss = preshift_init[-1]
    postshift_ss = postshift_init[-1]

    preshift_ss_gr = np.log(preshift_init[-1][0]/preshift_init[-2][0]) / dt
    postshift_ss_gr = np.log(postshift_init[-1][0]/postshift_init[-2][0]) / dt
    print(postshift_ss_gr)

    # Set the init params
    preshift_phiRb = preshift_ss[1] / preshift_ss[0]
    postshift_phiRb = postshift_ss[1] / postshift_ss[0]
    preshift_phiMb = 1 - preshift_phiRb
    preshift_TAA = preshift_ss[-2]
    preshift_TAA_star = preshift_ss[-1]

    init_params = [M0, M0 * preshift_phiRb, M0 *
                   preshift_phiMb, preshift_TAA, preshift_TAA_star]

    # time_range = np.arange(0, 10 / postshift_ss_gr, dt)
    time_range = np.arange(0, 100,  dt)
    preculture = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                                        init_params, time_range, args=postshift_args)

    preculture_df = pd.DataFrame(
        preculture, columns=['M', 'M_Rb', 'M_Mb', 'T_AA', 'T_AA_star'])
    preculture_df['time'] = time_range
    preculture_df['n_gen'] = time_range * postshift_ss_gr
    preculture_df['realized_phiRb'] = preculture_df['M_Rb'].values / \
        preculture_df['M']
    preculture_df['realized_phiMb'] = preculture_df['M_Mb'].values / \
        preculture_df['M']
    preculture_df['tRNA_balance'] = preculture_df['T_AA_star'].values / \
        preculture_df['T_AA']
    preculture_df['gamma'] = (7459/3600) * gamma_max * preculture_df['T_AA_star'].values / (preculture_df['T_AA_star'].values + Kd_TAA_star)
    preculture_df['prescribed_phiRb'] = preculture_df['tRNA_balance'].values / \
        (tau + preculture_df['tRNA_balance'].values)
    preculture_df['biomass_doublings'] = preculture_df['M'].values / M0
    preculture_df['nu_postshift'] = nu
    preculture_df['postshift_gr'] = postshift_ss_gr
    preculture_df['ss_phiRb'] = postshift_phiRb
    preculture_df['ss_diff'] = preculture_df['realized_phiRb'].values - postshift_phiRb
    growth_rate = np.log(preculture_df['M'].values[1:]/preculture_df['M'].values[:-1]) / dt
    growth_rate = np.append(growth_rate, postshift_ss_gr) 
    preculture_df['growth_rate'] = growth_rate

    dfs.append(preculture_df)
    err_df = err_df.append({'time': t_stop/dt,
                            'mass_frac': preculture_df['realized_phiRb'].values[int(t_stop/dt)],
                            'error': preculture_df['ss_diff'].values[int(t_stop/dt)],
                            'growth_rate_hr': postshift_ss_gr},
                            ignore_index=True)

preculture_df = pd.concat(dfs)
# %%
fig, ax = plt.subplots(3, 2, figsize=(6, 7))
ax[2,0].set_yscale('log')

for g, d in preculture_df.groupby(['postshift_gr']):
    ax[0, 0].plot(d['time'],
              d['prescribed_phiRb'], '-', lw=1, label=f'{g:0.2f} per hr')
    ax[0, 1].plot(d['time'],
              d['realized_phiRb'], '-', lw=1, label=f'{g:0.2f} per hr')
    ax[1, 0].plot(d['time'],
              d['tRNA_balance'], '-', lw=1, label=f'{g:0.2f} per hr')
    ax[1, 1].plot(d['time'],
              d['gamma'], '-', lw=1, label=f'{g:0.2f} per hr')
    ax[2, 1].plot(d['time'], d['growth_rate'], '-', lw=1)
    ax[2, 0].plot(d['time'], d['M'].values / M0, '-',  lw=1)
    
        # ax[0, i].hlines(postshift_phiRb, 0, preculture_df['time'].max(), color=colors['primary_red'], linestyle=':',
                    # lw=1, zorder=1000)
        # ax[1, i].hlines(postshift_phiRb, 0, 10, color=colors['primary_red'], linestyle=':',
                    # lw=1, zorder=1000)

for a in ax.ravel():
    a.set_xlabel('preculture growth time [hr]')

ax[0,  0].set_ylabel('$\phi_{Rb}$')
ax[0,  1].set_ylabel('$M_{Rb}/M$')
ax[1,  0].set_ylabel('charged tRNA / uncharged tRNA')
ax[1,  1].set_ylabel('translation rate [AA/s]')
ax[2,  0].set_ylabel('relative biomass')
ax[2,  1].set_ylabel('instantaneous growth rate [per hr]')

ax[0, 0].legend(title='postshift growth rate')
plt.savefig(f'../../figures/downshift_timescales.pdf', bbox_inches='tight')
# %%
fig, ax = plt.subplots(1, 1)
for g, d in preculture_df.groupby(['postshift_gr']):
    ax.plot(d['n_gen'], d['ss_diff'], '-', lw=1, label=f'{g:0.2f} per hr')
leg = ax.legend(title='postshift growth rate', fontsize=8)
leg.get_title().set_fontsize(8)
ax.set_xlim([0, 6])
ax.set_ylabel('$(M_{Rb}/M) - \phi_{Rb, ss}$', fontsize=12)
ax.set_xlabel('preculture growth time [generations]', fontsize=12)
ax.set_yscale('log')
ax.set_ylim([5E-4, 0.5])
ax.set_yticks([1E-3, 1E-2, 1E-1])
ax.set_yticklabels(['0.1%', '1%', '10%'], fontsize=8)
ax.xaxis.set_tick_params(labelsize=8)
plt.tight_layout()
plt.savefig('../../figures/downshift_ss_gen.pdf', bbox_inches='tight')
# %%
dai = dai[dai['growth_rate_hr'] > 0]
dai['corrected'] = dai['mass_fraction'].values - err_df['error'].values
dai
# %%
fig, ax = plt.subplots(1, 1)
ax.plot(dai['growth_rate_hr'], dai['mass_fraction'], 'o', label='Reported Value')
ax.plot(dai['growth_rate_hr'], dai['corrected'], 'o', label='Corrected Value')
ax.plot(opt_mu_ecoli, opt_phiRb_ecoli, 'k-', label='theory')
# %%
