#%% 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.model 
import growth.viz
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']
phi_O = 0.55
nu_max = 5 

# Set the initial conditions
init_phiRb = 0.001
init_phiMb = 1 - phi_O - init_phiRb
M0 = 1
M_Rb = init_phiRb * M0
M_Mb = init_phiMb * M0
TAA = 1E-5
TAA_star = 1E-5

init_params = [M0, M_Rb, M_Mb, TAA, TAA_star]
args = (gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O)
dt = 0.0001
time = np.arange(0, 5, dt)
out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                            init_params, time, args=args)
df = pd.DataFrame(out, columns=['M', 'M_Rb',  'M_Mb', 'TAA', 'TAA_star'])
df['ratio'] = df['TAA_star'].values / df['TAA'].values
df['gamma'] = gamma_max * (df['TAA_star'].values / (df['TAA_star'].values + Kd_TAA_star))
df['nu'] = nu_max * (df['TAA'].values / (df['TAA'].values + Kd_TAA))
df['Nu'] = df['nu'].values * df['M_Mb'].values
df['Gamma'] = df['gamma'].values * df['M_Rb'].values 
df['phi_Rb'] = df['ratio'].values / (df['ratio'].values + tau)
df['time'] = time
#%%
fig, ax = plt.subplots(3, 1, sharex=True, figsize=(6,6))
ax[0].set_ylabel('relative biomass')
ax[1].set_ylabel('phiRb')
ax[2].set_ylabel(r'$N/\Gamma$')
ax[2].set_xlabel('time [hr]')
ax[0].plot(df['time'], df['M'] / M0)
ax[1].plot(df['time'], df['phi_Rb'])
ax[2].plot(df['time'], df['Nu'].values / df['Gamma'].values)
# ax[2].set_yscale('log')
ax[0].set_yscale('log')
ax[2].set_ylim([0.9, 1.1])
# ax[2].set_xlim([0, 0.01])
# %%


phi_O = 0.55
phiRb_range = np.linspace(0.001, 0.45-0.001, 300)
nu_max = np.linspace(0, 10, 10) 
for i, n in enumerate(nu_max):
    lam = growth.model.steady_state_growth_rate(gamma_max, phiRb_range, n, const['Kd_cpc'], phi_O)
    gamma = growth.model.steady_state_gamma(gamma_max, phiRb_range, n, const['Kd_cpc'], phi_O)
    opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, n, const['Kd_cpc'], phi_O)
    opt_cAA = growth.model.steady_state_precursors(gamma_max, opt_phiRb, n, const['Kd_cpc'], phi_O)
    opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, n, const['Kd_cpc'], phi_O)
    Gamma = gamma_max * phiRb_range
    Nu = n * (1 - phiRb_range - phi_O)
    # Plot the tent
    plt.plot(phiRb_range, lam, label='lambda')
    # plt.plot(phiRb_range, Nu, '--',  label='Nu')
    # plt.plot(phiRb_range, Gamma, '--', label='Gamma')
    parity = n * (1 - phi_O) / (gamma_max * (opt_cAA + 1) + n)
    parity_gamma = growth.model.steady_state_gamma(gamma_max, parity, n, const['Kd_cpc'], phi_O)
    parity_lam = parity_gamma * parity
    plt.plot(opt_phiRb, opt_lam, 'ko')
    plt.plot(parity, parity_lam, 'ro')
    print(opt_phiRb - parity)
plt.legend()
# plt.ylim(0, 1)

# Plot the parity


# %%
plt.plot(phiRb_range, Gamma/Nu)
plt.vlines(opt_phiRb, 1E-3, 1E3)
plt.plot((1 - phi_O) / (1 + gamma_max/nu_max), 1, 'o')
plt.yscale('log')
# Gamma

# %%
import sympy as sp
# %%
sp.init_printing()
gamma_max = sp.Symbol('{{\gamma_{max}}}')
nu_max = sp.Symbol(r'{{\nu_{max}}}')
phiRb = sp.Symbol('{{\phi_{Rb}}}')
phiO = sp.Symbol('{{\phi_O}}')
Kd = sp.Symbol('{{K_D}}')
alpha = nu_max * (1 - phiO - phiRb) / (gamma_max * phiRb)

cAA = alpha - 1 + sp.sqrt((1 - alpha)**2 - 4 * Kd)

eq = nu_max * (1 - phiO) / (gamma_max + cAA + nu_max)
soln = sp.solve(eq - phiRb, phiRb)

# %%
phi_O = 0.55
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
nu_max = 4 
Kd_TAA = const['Kd_TAA']
tau = const['tau']
alpha = 10
a = (alpha / tau) * gamma_max 
b = (nu_max - kappa_max)
c = -nu_max * (1 - phi_O) 

ppGpp_phiR = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)
opt_phiR = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)

# %%
print(ppGpp_phiR)
print(opt_phiR)

# %%
nu_max_range = np.linspace(0.1, 15, 100)
alpha = 15
a = (alpha / tau) * gamma_max 
b = (nu_max_range - kappa_max)
c = -nu_max_range * (1 - phi_O) 
ppGpp_phiRb = (-b + np.sqrt(b**2 - 4 * a * c)) / (2 * a)

opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max_range, Kd_cpc, phi_O)

plt.plot(nu_max_range, opt_phiRb, '-', color=colors['primary_blue'])
plt.plot(nu_max_range, ppGpp_phiRb, '--', color=colors['primary_red'])
plt.xlabel('metabolic rate [per hr]')
plt.ylabel('optimum phiRb')
# %%
phiRb_range = np.linspace(0.001, (1 - phi_O) - 0.001, 100)
nu_max = 8 
phi_Mb = (1 - phi_O - phiRb_range)
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
opt_caa = growth.model.steady_state_precursors(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O)
cAA = opt_caa * 0.1
lam = growth.model.steady_state_growth_rate(gamma_max, phiRb_range, nu_max, Kd_cpc, phi_O)
gamma= gamma_max * cAA / (cAA + Kd_cpc)
plt.plot(phiRb_range, lam, '-', label='$\lambda$')
plt.plot(phiRb_range, nu_max * phi_Mb, '--', label='metabolic output rate')
plt.plot(phiRb_range, gamma * phiRb_range * (1 + cAA), '--', label='translational output rate + dilution rate')
plt.vlines(opt_phiRb, 0, 2.5, 'r', label='optimum $\phi_{Rb}$')
plt.ylim([0, 2.5])
plt.legend()
plt.xlabel('$\phi_{Rb}$')
plt.ylabel('rate [per hr]')
# %%
