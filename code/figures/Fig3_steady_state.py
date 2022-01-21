#%%
import numpy as np 
import matplotlib.pyplot as plt 
import growth.model 
import growth.viz 
import seaborn as sns
colors, _  = growth.viz.matplotlib_style()
const = growth.model.load_constants()
const['Kd_cpc'] = 0.03
# Define the parameter ranges
nu_max = np.array([0, 0.462, 0.925, 1.39, 1.85, 2.48, 3.11, 3.74, 4.37, 5]) 
nu_max *= (4.5 / 1.85) # Adjusting for the change in phi_O
phi_O = 0.55
phiRb_range = np.linspace(0.001, 1 - phi_O - 0.001, 300)

palette = sns.color_palette('bone', len(nu_max) + 2)

# generate the figure
fig, ax = plt.subplots(2, 3, figsize=(7, 4.25))
for i in range(3):
    ax[0, i].set_xlabel('allocation towards ribosomes\n' + ' $\phi_{Rb}$', fontsize=8)
    ax[1, i].set_xlabel('metabolic rate\n' +  r' $\nu_{max}$ [hr$^{-1}$]', fontsize=8)
    ax[0, i].set_xlim([0, 0.45])

# Add axis labels
ax[0, 0].set(ylabel=r'$c_{pc}^* / K_D^{c_{pc}}$' + '\nprecursor concentration', yscale='log')
ax[1, 0].set_ylabel('$\phi_{Rb}$' + '\nallocation towards ribosomes')
for i in range(2):
        ax[i,1].set(ylabel='$\gamma / \gamma_{max}$' + '\nrelative translation rate')
        ax[i,2].set(ylabel='$\lambda$ [hr$^{-1}$]' + '\ngrowth rate')
ax[1, 1].set_ylim([0, 1])

# Add panel values
height=0.89
for i, nu in enumerate(nu_max):
    # Compute the steady state values
    params = (const['gamma_max'], phiRb_range, nu, const['Kd_cpc'], phi_O)

    lam = growth.model.steady_state_growth_rate(*params)
    gamma = growth.model.steady_state_gamma(*params)
    cpc = growth.model.steady_state_precursors(*params)

    # Add glyphs
    if nu == nu_max[4]:
        print(nu)
        color = colors['primary_red']
        zorder = 1000
        lw = 1.5
    else:
        color = palette[i]
        zorder = 1
        lw = 1
    ax[0,0].plot(phiRb_range, cpc / const['Kd_cpc'], color=color, lw=lw, zorder=zorder)
    ax[0,1].plot(phiRb_range, gamma / const['gamma_max'], color=color, lw=lw, zorder=zorder)
    ax[0,2].plot(phiRb_range, lam, color=color, lw=lw, zorder=zorder)

    # Plot the maximum 
    opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'], nu, 
                                                      const['Kd_cpc'], phi_O)
    opt_lam = growth.model.steady_state_growth_rate(const['gamma_max'],
                                    opt_phiRb, nu, const['Kd_cpc'], phi_O)
    opt_gamma = growth.model.steady_state_gamma(const['gamma_max'], opt_phiRb,
                                        nu, const['Kd_cpc'], phi_O)
    opt_cpc = growth.model.steady_state_precursors(const['gamma_max'],
                                opt_phiRb, nu, const['Kd_cpc'], phi_O)
    ax[0,0].plot(opt_phiRb, opt_cpc / const['Kd_cpc'], 'o', ms=4, color=color, zorder=1001)
    ax[0,1].plot(opt_phiRb, opt_gamma / const['gamma_max'], 'o', ms=4, color=color, zorder=1001)
    ax[0,2].plot(opt_phiRb, opt_lam, 'o', ms=4, color=color, zorder=1001)

# Ccompute the various scenarios
nu_max = np.linspace(0, 15, 300)

# Scenario II: Constant gamma
cpc_Kd = 10
phiRb_trans = growth.model.phiRb_constant_translation(const['gamma_max'], nu_max, cpc_Kd, const['Kd_cpc'], phi_O)
strat2_params = (const['gamma_max'], phiRb_trans, nu_max, const['Kd_cpc'], phi_O)
phiRb_trans_lam = growth.model.steady_state_growth_rate(*strat2_params)
phiRb_trans_gamma = growth.model.steady_state_gamma(*strat2_params)
ax[1, 0].plot(nu_max, phiRb_trans, lw=1, color=colors['primary_green'])
ax[1, 1].plot(nu_max, phiRb_trans_gamma / const['gamma_max'], lw=1, color=colors['primary_green'])
ax[1, 2].plot(nu_max, phiRb_trans_lam, lw=1, color=colors['primary_green'])


# Scenario III: Optimal ribosomal allocation
phiRb_opt = growth.model.phiRb_optimal_allocation(const['gamma_max'], nu_max, const['Kd_cpc'], phi_O)
strat3_params = (const['gamma_max'], phiRb_opt, nu_max, const['Kd_cpc'], phi_O) 
phiRb_opt_lam = growth.model.steady_state_growth_rate(*strat3_params)
phiRb_opt_gamma = growth.model.steady_state_gamma(*strat3_params)
ax[1, 0].plot(nu_max, phiRb_opt, lw=1, color=colors['primary_blue'])
ax[1, 1].plot(nu_max, phiRb_opt_gamma / const['gamma_max'], lw=1, color=colors['primary_blue'])
ax[1, 2].plot(nu_max, phiRb_opt_lam, lw=1, color=colors['primary_blue'])

# Scenario I: Constant allocation
phiRb_const = 0.2* np.ones_like(nu_max)
strat1_params = (const['gamma_max'],  phiRb_const, nu_max, const['Kd_cpc'], phi_O)
phiRb_const_lam = growth.model.steady_state_growth_rate(*strat1_params)
phiRb_const_cpc = growth.model.steady_state_precursors(*strat1_params)
phiRb_const_gamma = growth.model.steady_state_gamma(*strat1_params)
ax[1, 0].plot(nu_max, phiRb_const, lw=1, color=colors['primary_black'])
ax[1, 1].plot(nu_max, phiRb_const_gamma / const['gamma_max'], lw=1, color=colors['primary_black'])
ax[1, 2].plot(nu_max, phiRb_const_lam, lw=1, color=colors['primary_black'])

# Tighten and save
plt.tight_layout()
plt.savefig('../../figures/Fig3_steady_state_plots.pdf', bbox_inches='tight')

# %%
