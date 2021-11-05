#%%
import numpy as np 
import bokeh.plotting 
import pandas as pd
import bokeh.io
import bokeh.models
import growth.model
import scipy.integrate
import growth.viz
colors, palette = growth.viz.bokeh_style()
const = growth.model.load_constants()
bokeh.io.output_file('./interactive_upshift.html')


# Define the constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']

# Define the initial shift parameters
nu_preshift = 0.5
nu_postshift = 1.75
nu_range = np.arange(0.001, 5, 0.001)
phiO_preshift = 0.35
phiO_postshift = 0.25
time_shift = 3
time_end = 10
dt = 0.001
preshift_time = np.arange(0, time_shift, dt)
postshift_time = np.arange(time_shift, time_end, dt)

# Initial integration conditions
M0 = 1E9
TAA_0 = 0.002
TAA_star_0 = 0.002

# Figure out the initial conditions for the instantaneous and dynamic case
phiRb_preshift = growth.model.phiRb_optimal_allocation(gamma_max, nu_preshift, 
                                                    const['Kd_cpc'], phiO_preshift)
phiRb_postshift = growth.model.phiRb_optimal_allocation(gamma_max, nu_preshift, 
                                                    const['Kd_cpc'], phiO_preshift)
preshift_params = [M0, phiRb_preshift * M0, (1 - phiO_preshift - phiRb_preshift) * M0,  TAA_0, TAA_star_0]
postshift_params = [M0, phiRb_postshift * M0, (1 - phiO_postshift - phiRb_postshift) * M0,  TAA_0, TAA_star_0]
preshift_args = (gamma_max, nu_preshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_preshift, 0, True)
postshift_args = (gamma_max, nu_postshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_postshift, 0, True)
preshift_ss = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                preshift_params, np.arange(0, 100, 0.0001), args=preshift_args)
postshift_ss = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                postshift_params, np.arange(0, 100, 0.0001), args=postshift_args)
preshift_ss = preshift_ss[-1]
postshift_ss = postshift_ss[-1]
preshift_phiRb = preshift_ss[1] / preshift_ss[0]
preshift_phiMb = preshift_ss[2] / preshift_ss[0]
postshift_phiRb = postshift_ss[1] / postshift_ss[0]
postshift_phiMb = postshift_ss[2] / postshift_ss[0]

preshift_TAA = preshift_ss[3]
preshift_TAA_star = preshift_ss[4]


# Do the integration with dynamic reallocation 
init_params = [M0, preshift_phiRb * M0, preshift_phiMb * M0, preshift_TAA, preshift_TAA_star]
args = (gamma_max, nu_preshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_preshift, 0, True)
dynamic_preshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params, 
                                        preshift_time, args=args)
init_params = dynamic_preshift[-1]
args = (gamma_max, nu_postshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_postshift, 0, True)
dynamic_postshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params, 
                                        postshift_time, args=args)
dynamic_postshift = dynamic_postshift[1:]

# Do the integration with static reallocation 
init_params = [M0, preshift_phiRb * M0, preshift_phiMb * M0, preshift_TAA, preshift_TAA_star]
args = (gamma_max, nu_preshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_preshift, phiRb_preshift, False)
instant_preshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params, 
                                        preshift_time, args=args)
init_params = instant_preshift[-1]
args = (gamma_max, nu_postshift, tau, Kd_TAA, Kd_TAA_star, kappa_max, phiO_postshift, phiRb_postshift, False)
instant_postshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params, 
                                        postshift_time, args=args)
instant_postshift = dynamic_postshift[1:]



# Set up the data source
preshift_df = pd.DataFrame(dynamic_preshift, columns=['M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
postshift_df = pd.DataFrame(dynamic_postshift, columns=['M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
instant_df = pd.DataFrame(dynamic_preshift, columns=['M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
instant_df = pd.DataFrame(instant_postshift, columns=['M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])

preshift_df['time'] = preshift_time
postshift_df['time'] = postshift_time[1:]
dynamic_df = pd.concat([preshift_df, postshift_df], sort=False)
dynamic_df['balance'] = dynamic_df['TAA_star'].values / dynamic_df['TAA'].values
dynamic_df['phiRb'] = dynamic_df['balance'].values / (dynamic_df['balance'].values + tau) 
dynamic_df['Mrb_M'] = dynamic_df['Mrb'].values / dynamic_df['M'].values



dynamic_source = bokeh.models.ColumnDataSource(dynamic_df[['time', 'phiRb', 'Mrb_M', 'TAA', 'TAA_star']]) 


# ##############################################################################
# WIDGET DEFINITION
# ##############################################################################


# ##############################################################################
# CANVAS DEFINITION
# ##############################################################################
allocation_axis = bokeh.plotting.figure(width=350, height=350,
                                        x_axis_label = 'time')
