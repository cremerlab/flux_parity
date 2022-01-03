#%%
import numpy as np 
import matplotlib.pyplot as plt
import growth.model
import growth.viz
import growth.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# def f(u, u_star, nu_max, const):
#     phi_Rb = (1 - const['phi_O']) * (1 + (u * const['tau'])/u_star)**-1
#     phi_Mb = (1 - const['phi_O']) - phi_Rb
#     kappa = (const['kappa_max']/const['gamma_max']) * phi_Rb / (1 - const['phi_O'])
#     gamma = 1 / (1 + const['Kd_TAA_star'] * u_star**-1)
#     nu = (nu_max / const['gamma_max']) / (1 + const['Kd_TAA'] * u**-1)

#     du_dt = kappa + gamma * phi_Rb * (1 - u) - nu * phi_Mb
#     dustar_dt = nu * phi_Mb - gamma * phi_Rb * (1 + u_star)
#     return [du_dt, dustar_dt]
    

# Set the ranges
ranges = np.linspace(-6, 0, 75)
nu_max = 10
U, U_star = np.meshgrid(ranges, ranges)

u_out = np.zeros(U.shape)
ustar_out = np.zeros(U.shape)

args = {'kappa_max': const['kappa_max'],
                'gamma_max': const['gamma_max'],
                'Kd_TAA': const['Kd_TAA'],
                'Kd_TAA_star': const['Kd_TAA_star'],
                'nu_max': nu_max,
                'tau': const['tau'],
                'phi_O': const['phi_O']}
equil = growth.integrate.equilibrate_FPM(args)
#%%
for i, _u in enumerate(ranges):
    for j, _ustar in enumerate(ranges): 
        args = {'kappa_max': const['kappa_max'],
                'gamma_max': const['gamma_max'],
                'Kd_TAA': const['Kd_TAA'],
                'Kd_TAA_star': const['Kd_TAA_star'],
                'nu_max': nu_max,
                'tau': const['tau'],
                'phi_O': const['phi_O']}
        params = [1, equil[1]/equil[0], equil[2]/equil[0], 10**_u, 10**_ustar]
        _out = growth.model.self_replicator_FPM(params, 0, args) 
        u_out[i, j] = _out[-2]
        ustar_out[i, j] = _out[-1]
# %%
fig, ax = plt.subplots(1, 1)
# ax.quiver(U, U_star, u_out, ustar_out) 
ax.streamplot(U, U_star, u_out, ustar_out, density=2) 

# %%
