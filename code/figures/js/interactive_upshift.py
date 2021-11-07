# %%
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
phiRb_postshift = growth.model.phiRb_optimal_allocation(gamma_max, nu_postshift,
                                                        const['Kd_cpc'], phiO_postshift)
preshift_params = [M0, phiRb_preshift * M0,
                   (1 - phiO_preshift - phiRb_preshift) * M0,  TAA_0, TAA_star_0]
postshift_params = [M0, phiRb_postshift * M0,
                    (1 - phiO_postshift - phiRb_postshift) * M0,  TAA_0, TAA_star_0]
preshift_args = (gamma_max, nu_preshift, tau, Kd_TAA,
                 Kd_TAA_star, kappa_max, phiO_preshift)
postshift_args = (gamma_max, nu_postshift, tau, Kd_TAA,
                  Kd_TAA_star, kappa_max, phiO_postshift)
preshift_ss = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                     preshift_params, np.arange(0, 200, 0.0001), args=preshift_args)
postshift_ss = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                      postshift_params, np.arange(0, 200, 0.0001), args=postshift_args)
preshift_ss = preshift_ss[-1]
postshift_ss = postshift_ss[-1]
preshift_phiRb = preshift_ss[1] / preshift_ss[0]
preshift_phiMb = preshift_ss[2] / preshift_ss[0]
postshift_phiRb = postshift_ss[1] / postshift_ss[0]
postshift_phiMb = postshift_ss[2] / postshift_ss[0]

preshift_TAA = preshift_ss[3]
preshift_TAA_star = preshift_ss[4]


# Do the integration with dynamic reallocation
init_params = [M0, preshift_phiRb * M0,
               preshift_phiMb * M0, preshift_TAA, preshift_TAA_star]
args = (gamma_max, nu_preshift, tau, Kd_TAA,
        Kd_TAA_star, kappa_max, phiO_preshift, 0)
dynamic_preshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params,
                                          preshift_time, args=args)
init_params = dynamic_preshift[-1]
args = (gamma_max, nu_postshift, tau, Kd_TAA,
        Kd_TAA_star, kappa_max, phiO_postshift, 0)
dynamic_postshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params,
                                           postshift_time, args=args)
dynamic_postshift = dynamic_postshift[1:]

# Do the integration with static reallocation
init_params = [M0, preshift_phiRb * M0,
               preshift_phiMb * M0, preshift_TAA, preshift_TAA_star]
args = (gamma_max, nu_preshift, tau, Kd_TAA, Kd_TAA_star,
        kappa_max, phiO_preshift, phiRb_preshift, False, False)
instant_preshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params,
                                          preshift_time, args=args)
init_params = instant_preshift[-1]
args = (gamma_max, nu_postshift, tau, Kd_TAA, Kd_TAA_star,
        kappa_max, phiO_postshift, phiRb_postshift, False, False)
instant_postshift = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, init_params,
                                           postshift_time, args=args)
instant_postshift = instant_postshift[1:]


# Convert to dataframes (All this work for just the initialization plot!)
preshift_df = pd.DataFrame(dynamic_preshift, columns=[
                           'M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
postshift_df = pd.DataFrame(dynamic_postshift, columns=[
                            'M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
instant_df_preshift = pd.DataFrame(instant_preshift, columns=[
                          'M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
instant_df_postshift = pd.DataFrame(instant_postshift, columns=[
                          'M', 'Mrb', 'Mmb', 'TAA', 'TAA_star'])
instant_df_preshift['phiRb'] = preshift_phiRb
instant_df_postshift['phiRb'] = postshift_phiRb
preshift_df['time'] = preshift_time
postshift_df['time'] = postshift_time[1:]
instant_df_preshift['time'] = preshift_time
instant_df_postshift['time'] = postshift_time[1:]

dynamic_df = pd.concat([preshift_df, postshift_df], sort=False)
instant_df = pd.concat([instant_df_preshift, instant_df_postshift], sort=False)
dynamic_df['balance'] = dynamic_df['TAA_star'].values / \
    dynamic_df['TAA'].values
dynamic_df['phiRb'] = dynamic_df['balance'].values / \
    (dynamic_df['balance'].values + tau)
dynamic_df['Mrb_M'] = dynamic_df['Mrb'].values / dynamic_df['M'].values
instant_df['Mrb_M'] = instant_df['Mrb'].values / instant_df['M'].values
dynamic_df['total_tRNA'] = (dynamic_df['TAA'].values / Kd_TAA) + (dynamic_df['TAA_star'].values / Kd_TAA_star)
instant_df['total_tRNA'] = (instant_df['TAA'].values / Kd_TAA) + (instant_df['TAA_star'].values / Kd_TAA_star)
dynamic_gr = np.log(dynamic_df['M'].values[1:] / dynamic_df['M'].values[:-1]) / dt
instant_gr = np.log(instant_df['M'].values[1:] / instant_df['M'].values[:-1]) / dt

# Set up the data sources
source = bokeh.models.ColumnDataSource({
        'time' : [instant_df['time'].values - time_shift, dynamic_df['time'].values - time_shift],
        'phiRb': [instant_df['phiRb'].values, dynamic_df['phiRb'].values],
        'Mrb_M': [instant_df['Mrb_M'].values, dynamic_df['Mrb_M'].values],
        'tRNA' : [instant_df['total_tRNA'].values, dynamic_df['total_tRNA'].values], 
        'lam'  : [instant_gr, dynamic_gr],
        'lam_time' : [instant_df['time'].values[:-1] - time_shift, dynamic_df['time'].values[:-1] - time_shift],
        'color' : [colors['primary_blue'], colors['primary_red']],
        'label' : ['instantaneous reallocation', 'dynamic reallocation via ppGpp'],
        'filler_xs' : [[], []],
        'filler_ys' : [[], []]})


# ##############################################################################
# WIDGET DEFINITION
# ##############################################################################
nu_preshift_slider = bokeh.models.Slider(start=0.001, end=5, step=0.001, value=nu_preshift,
                                        title='preshift metabolic rate [inv. hr]')
nu_postshift_slider = bokeh.models.Slider(start=0.001, end=5, step=0.001, value=nu_postshift,
                                        title='postshift metabolic rate [inv. hr]')
phiO_preshift_slider = bokeh.models.Slider(start=0.001, end=5, step=0.001, value=phiO_preshift,
                                        title="preshift allocation to 'other' proteins")
phiO_postshift_slider = bokeh.models.Slider(start=0.001, end=5, step=0.001, value=phiO_postshift,
                                        title="postshift allocation to 'other' proteins")

# ##############################################################################
# CANVAS DEFINITION
# ##############################################################################
allocation_axis = bokeh.plotting.figure(width=350, height=350,
                                        x_axis_label='time from shift [hr]',
                                        y_axis_label=r'$$\phi_{Rb}$$',
                                        title='ribosomal allocation')

ribocontent_axis = bokeh.plotting.figure(width=350, height=350,
                                        x_axis_label='time from shift [hr]',
                                        y_axis_label=r'$$\frac{M_{Rb}}{M}$$',
                                        title='ribosomal content')
precursor_axis = bokeh.plotting.figure(width=350, height=350,
                                        x_axis_label='time from shift [hr]',
                                        y_axis_label=r'$$\frac{tRNA}{K_D^{tRNA}} + \frac{tRNA^*}{K_D^{tRNA^*}}$$',
                                        title='total tRNA abundance',
                                        y_axis_type='log')
growth_axis = bokeh.plotting.figure(width=350, height=350,
                                        x_axis_label='time from shift [hr]',
                                        y_axis_label=r'λ [inv. hr]',
                                        title='instantaneous growth rate')

# ##############################################################################
# GLYPH DEFINITION
# ##############################################################################
allocation_axis.multi_line(xs='time', ys='phiRb', line_color='color', line_width=2,
                           source=source)
ribocontent_axis.multi_line(xs='time', ys='Mrb_M', line_color='color', line_width=2,
                           source=source)
precursor_axis.multi_line(xs='time', ys='tRNA', line_color='color', line_width=2,
                           source=source)
growth_axis.multi_line(xs='time', ys='lam', line_color='color', line_width=2,
                           source=source)
# ##############################################################################
# CALLBACK DEFINITION
# ##############################################################################
args = {'phiO_preshift_slider': phiO_preshift_slider,
        'phiO_postshift_slider': phiO_postshift_slider,
        'nu_max_preshift_slider': nu_preshift_slider,
        'nu_max_postshift_slider': nu_postshift_slider,
        'gamma_max': gamma_max,
        'Kd_TAA' : Kd_TAA,
        'Kd_TAA_star': Kd_TAA_star,
        'kappa_max': kappa_max,
        'preshift_time': preshift_time,
        'postshift_time': postshift_time,
        'tau': tau,
        'source': source} 
callback = growth.viz.load_js(['./interactive_upshift.js', './functions.js'],
                        args=args)
for s in [phiO_preshift_slider, phiO_postshift_slider, nu_preshift_slider, nu_postshift_slider]:
    s.js_on_change('value', callback)

# ##############################################################################
# LAYOUT
# ##############################################################################
slider1 = bokeh.layouts.Column(nu_preshift_slider, nu_postshift_slider)
slider2 = bokeh.layouts.Column(phiO_preshift_slider, phiO_postshift_slider)
sliders = bokeh.layouts.Row(slider1, slider2)
plots1 = bokeh.layouts.Row(allocation_axis, ribocontent_axis)
plots2 = bokeh.layouts.Row(precursor_axis, growth_axis)
layout = bokeh.layouts.Column(sliders, plots1, plots2)
bokeh.io.save(layout)


# %%
