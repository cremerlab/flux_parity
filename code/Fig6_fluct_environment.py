#%%
import numpy as np 
import matplotlib.pyplot as plt
import growth.viz
import growth.model
import growth.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

# Set the constants
gamma_max = const['gamma_max']
nu_init = 1 
nu_shift = 3 
total_time = 10 
shift_time = 2
phi_O = 0.65
phi_O_post = 0.55 
dt = 0.0001
total_time = 8 
shift_time = 2

# ppGpp params
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau'] 
kappa_max = const['kappa_max']

#%% Equilibrate the pre and postshift conditions
preshift_args = {'gamma_max':gamma_max,
                 'nu_max':nu_init,
                 'Kd_TAA':Kd_TAA,
                 'Kd_TAA_star':Kd_TAA_star,
                 'kappa_max':kappa_max,
                 'tau':tau,
                 'phi_O':phi_O}
postshift_args = {'gamma_max':gamma_max,
                 'nu_max':nu_shift,
                 'Kd_TAA':Kd_TAA,
                 'Kd_TAA_star':Kd_TAA_star,
                 'kappa_max':kappa_max,
                 'tau':tau,
                 'phi_O':phi_O_post}


# Equilibrate to find the postshift allocation
preshift = growth.integrate.equilibrate_FPM(preshift_args)
postshift = growth.integrate.equilibrate_FPM(postshift_args)
post_phiRb = postshift[1]/postshift[0]
pre_phiRb = preshift[1]/preshift[0]

# Do the shift
dynamic_shift = growth.integrate.nutrient_shift_FPM([preshift_args, postshift_args],
                                                    total_time=total_time,
                                                    shift_time=shift_time,
                                                    dt=dt)
dynamic_shift['ratio'] = dynamic_shift['TAA_star'].values / dynamic_shift['TAA']
dynamic_shift['phi_O'] = phi_O
dynamic_shift.loc[dynamic_shift['shifted_time']>= 0, 'phi_O'] =  phi_O_post
dynamic_shift['phiRb'] = (1 - dynamic_shift['phi_O'].values) *\
                        (dynamic_shift['ratio'].values / (dynamic_shift['ratio'].values + tau))
gr = list(np.log(dynamic_shift['M'].values[1:]/dynamic_shift['M'].values[:-1]) / dt)
gr.append(gr[-1])
dynamic_shift['lam'] = gr

# Perform the instant shift
preshift_args['phiRb'] = pre_phiRb
postshift_args['phiRb'] = post_phiRb

inst_shift = growth.integrate.nutrient_shift_FPM([preshift_args, postshift_args],
                                                total_time=total_time,
                                                shift_time=shift_time,
                                                dt=dt)
inst_shift['phiRb'] = pre_phiRb 
inst_shift.loc[inst_shift['shifted_time'] >= 0, 'phiRb'] = post_phiRb
gr = list(np.log(inst_shift['M'].values[1:]/inst_shift['M'].values[:-1]) / dt)
gr.append(gr[-1])
inst_shift['lam'] = gr


#%% Set up the figure canvas
fig, ax = plt.subplots(1, 2, figsize=(4.25, 2)) 

# Format and label axes
for a in ax:
    a.set_xlabel('time from upshift [hr]')
ax[0].set_ylabel('$\phi_{Rb}$\nallocation towards ribosomes')
ax[1].set_ylabel('$\lambda_i$ [hr$^{-1}$]\ninstantaneous growth rate')
ax[0].set_ylim([0, 0.5])
ax[1].set_ylim([0.1, 1])

# Plot the instantaneous reallocation
ax[0].plot(inst_shift['shifted_time'], inst_shift['phiRb'], '-', lw=1,
           color=colors['primary_blue'])
ax[1].plot(inst_shift['shifted_time'], inst_shift['lam'], '-', lw=1,
           color=colors['primary_blue'])

# Plot the flux-parity solution
ax[0].plot(dynamic_shift['shifted_time'], dynamic_shift['phiRb'], '--', lw=1,
           color=colors['primary_red'])
ax[1].plot(dynamic_shift['shifted_time'], dynamic_shift['lam'], '--', lw=1,
           color=colors['primary_red'])
           
plt.tight_layout()
plt.savefig('../figures/main_text/Fig6_upshift_plots.pdf', bbox_inches='tight')
#%%
