#%%
import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 
import bokeh.io
import bokeh.plotting 
import bokeh.models 
import bokeh.layouts
import growth.viz 
import growth.integrate
import growth.model
import tqdm
colors, palette = growth.viz.bokeh_style()
const = growth.model.load_constants()

# Load the reference parameter set
gamma_max = const['gamma_max']
nu_max = 4.5
tau = const['tau']
kappa_max = const['kappa_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
phi_O = const['phi_O']

# Find the equilibrium state
args = {'gamma_max': gamma_max,
        'nu_max': nu_max,
        'kappa_max':kappa_max,
        'tau':tau,
        'Kd_TAA': Kd_TAA,
        'Kd_TAA_star':Kd_TAA_star,
        'phi_O': phi_O}
out = growth.integrate.equilibrate_FPM(args)

# Unpack equilibrium parameters
TAA = out[-2]
TAA_star = out[-1]
ratio = TAA_star / TAA 
gamma_equil = gamma_max * TAA_star / (TAA_star + Kd_TAA_star)
nu_equil = nu_max * TAA / (TAA + Kd_TAA)
phiRb = (1 - phi_O) * ratio / (ratio + tau)

kappa = kappa_max * ratio / (ratio + tau)
phiRb_range = np.arange(0, 1 - phi_O, 0.001)

#%%
dt = 0.00001
df = pd.DataFrame({})
for i, phi in enumerate(tqdm.tqdm(phiRb_range)):
    # Equilibrate the model at this phiR
    args = {'gamma_max': gamma_max,
            'nu_max': nu_max,
            'Kd_TAA': Kd_TAA,
            'Kd_TAA_star': Kd_TAA_star,
            'kappa_max': kappa_max,
            'phi_O': phi_O,
            'tau': tau,
            'dynamic_phiRb': False,
            'phiRb':  phi}
    _out = growth.integrate.equilibrate_FPM(args, t_return=2)
    out = _out[-1]
    lam = np.log(_out[-1][0] / _out[0][0]) / dt

    # Compute the various properties
    gamma = gamma_max * (out[-1] / (out[-1] + Kd_TAA_star))
    nu = nu_max * (out[-2] / (out[-2] + Kd_TAA))
    ratio = out[-1] / out[-2]

    results = {'gamma': gamma,
               'nu': nu, 
               'lam': gamma * phi, 
               'TAA': out[-2],
               'TAA_star':out[-1],
               'ratio': out[-1]/out[-2],
               'tot_tRNA': out[-1] + out[-2],
               'kappa': kappa_max * ratio / (ratio + tau),
               'phi_Rb': phi,
               'metabolic_flux': nu * (1 - phi_O - phi),
               'translational_flux': gamma * phi}
    df = df.append(results, ignore_index=True)

#%%
bokeh.io.output_file('./interactive_equilibration.html')
dt = 0.00001
time_range = np.arange(0, 10, dt)
# ##############################################################################
# SOURCE DEFINITION
# ##############################################################################
perturbed_point = bokeh.models.ColumnDataSource({'phiRb':[phiRb],
                                                 'growth_rate': [gamma_equil * phiRb]})
fluxes = bokeh.models.ColumnDataSource({'phiRb_range': list(phiRb_range),
                                        'metab_flux':list(nu_equil * (1 - phiRb_range - phi_O)),
                                        'trans_flux': list(gamma_equil * phiRb_range )})

# ##############################################################################
# WIDGET DEFINITION
# ##############################################################################
phiRb_slider = bokeh.models.Slider(start=phiRb_range[0], end=phiRb_range[-1], step=np.diff(phiRb_range)[0],
                                    value=phiRb, title='allocation towards ribosomes')

# ##############################################################################
# CANVAS DEFINITION
# ##############################################################################
tent_ax = bokeh.plotting.figure(width=500, 
                                height=500, 
                                x_axis_label = 'allocation towards ribosomes',
                                y_axis_label = 'rate [per hr]',
                                y_range=[0, 1.75])

# ##############################################################################
# GLYPH DEFINITION
# ##############################################################################
tent_ax.line(x='phi_Rb', y='lam',  color=colors['primary_black'], line_width=2,
             legend_label='growth rate λ', source=df)
tent_ax.line(x='phiRb_range', y='metab_flux', color=colors['primary_purple'],
            line_width=2, source=fluxes, line_dash='dashed', legend_label='metabolic flux')
tent_ax.line(x='phiRb_range', y='trans_flux', color=colors['primary_gold'],
            line_width=2, source=fluxes, line_dash='dashed',
            legend_label='translational flux')
tent_ax.circle(x='phiRb', y='growth_rate', size=10, color=colors['primary_black'],
             source=perturbed_point)

# ##############################################################################
# CALLBACK DEFINITION
# ##############################################################################
init_args = {'phiRb_range': phiRb_range,
             'growth_rate_range': df['lam'].values,
             'kappa': df['kappa'].values,
             'gamma': df['gamma'].values,
             'nu': df['nu'].values,
             'kappa': df['kappa'].values,
             'tot_tRNA': df['tot_tRNA'].values,
             'flux_source':fluxes,
             'point_source': perturbed_point,
             'phiRb_slider': phiRb_slider,
             'phi_O': phi_O} 

init_cb = growth.viz.load_js('./interactive_equilibration_init.js', init_args)
phiRb_slider.js_on_change('value', init_cb)


# ##############################################################################
# LAYOUT DEFINITION
# ##############################################################################
layout = bokeh.layouts.Column(phiRb_slider, tent_ax)
bokeh.io.save(layout)
# %%
