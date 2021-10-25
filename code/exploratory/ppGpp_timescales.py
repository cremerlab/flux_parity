#%%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 
import scipy.integrate
import growth.model
import growth.viz
import seaborn as sns
colors, _= growth.viz.matplotlib_style()


# Set the constants
gamma_max = 20 * 3600/ 7459 
nu_init = 0.535
nu_shift = 1.05 #1.83
total_time = 8
shift_time = 2

# ppGpp params
Kd_TAA = 2E-5
Kd_TAA_star = 2E-5 
tau = 3
phi_Rb = 0.5
phi_Mb = 1 - phi_Rb
OD_CONV = 1.5E17


# Init params
M0 = 0.001 * OD_CONV
M_Rb = phi_Rb * M0
M_Mb = phi_Mb * M0
T_AA = 0.0002
T_AA_star = 0.0002
kappa_max = (88 * 5 * 3600) / 1E9 #0.002
dt = 0.0001

init_params = [M_Rb, M_Mb, T_AA, T_AA_star]
preshift_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
postshift_args = (gamma_max, nu_shift, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
preshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=preshift_args)
postshift_out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator_ppGpp,
                             init_params, np.arange(0, 200, dt), args=postshift_args)

preshift_out = preshift_out[-1]
postshift_out = postshift_out[-1]
init_phiRb = (preshift_out[0]) / (preshift_out[0] + preshift_out[1])
shift_phiRb = (postshift_out[0]) / (postshift_out[0] + postshift_out[1])
init_phiMb = 1 - init_phiRb
shift_phiMb = 1 - shift_phiRb
init_T_AA = preshift_out[2]
init_T_AA_star = preshift_out[3]

# Do the shift
init_params = [M0 * init_phiRb, M0 * init_phiMb, init_T_AA, init_T_AA_star]
init_args = (gamma_max, nu_init, tau, Kd_TAA_star, Kd_TAA, False, True, True, kappa_max)
# %%
