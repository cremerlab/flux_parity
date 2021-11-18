#%%
import sympy as sp

nu_max = sp.Symbol(r'{{\nu_{max}}}')
gamma_max = sp.Symbol('{{\gamma_{max}}}')
phi_Rb = sp.Symbol('{{\phi_{Rb}}}')
phi_O = sp.Symbol('{{\phi_O}}')
phi_Mb = 1 - phi_Rb - phi_O
Kd_TAA = sp.Symbol('{{K_D}}')
Kd_TAA_star = sp.Symbol('{{K_{D}^*}}')
Kd = sp.Symbol('{{K_D}}')
TAA = sp.Symbol('{{tRNA}}')
TAA_star = sp.Symbol('{{tRNA^*}}')
kappa = sp.Symbol('{{\kappa}}')
gamma = gamma_max * TAA_star / (TAA_star + Kd)
nu = nu_max *  TAA / (TAA + Kd)
theta = sp.Symbol(r'{{\theta}}')
tau = sp.Symbol(r'{{\tau}}')
eq = nu * phi_Mb - gamma * phi_Rb * (1 + TAA_star) + nu * phi_Mb - kappa - gamma * phi_Rb * (1 - TAA)

# %%
out = sp.solve(eq, phi_Rb)

# %%
numer = (1 + tau / theta) * (Kd * theta * TAA) * (2 * nu_max * TAA * (1 - phi_O) - kappa * (Kd + TAA))
denom = gamma_max * Kd * theta * TAA**2 * (1 - theta) + 2 * Kd * nu_max * TAA * (1 + theta) + gamma_max * TAA**3 * theta * (theta - 1) + 2 * TAA**2 * theta * (gamma_max + nu_max)
theta_eq = 1 - (numer / denom)
# %%
theta_soln = sp.solve(theta_eq, theta)

# %%

