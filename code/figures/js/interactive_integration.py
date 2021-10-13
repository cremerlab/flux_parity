#%%
import numpy as np 
import pandas as pd
import scipy.integrate
import bokeh.models 
import bokeh.plotting 
import bokeh.layouts
import bokeh.io 
import growth.model
import growth.viz
bokeh.io.output_file('./interactive_integration.html')
colors, palette = growth.viz.bokeh_style() 

# Set starting values for params
gamma_max = 9.65
nu_max = 5
omega = 3E20
Kd_cAA = 0.025
Kd_cN = 5E-4
phi_R = 0.2

# Set initial conditions
OD_CONV = 1.5E17
M0 = 0.01 * OD_CONV
MR_0 = phi_R * M0
MP_0 = (1 - phi_R) * M0
cAA_0 = 1E-3 
cN_0 = 0.010

# Perform the integration and instantiate data source
time_range = np.linspace(0, 20, 3000)
params = [MR_0, MP_0, cAA_0, cN_0]
args = (gamma_max, nu_max, omega, phi_R, Kd_cAA, Kd_cN)
out = scipy.integrate.odeint(growth.model.batch_culture_self_replicator,
                            params, time_range, args=args)
df = pd.DataFrame(out, columns=['Mr', 'Mp', 'cAA', 'cN'])
df['M'] = df['Mr'].values + df['Mp'].values
df['rel_M'] = df['M'].values / M0
df['cAA_Kd'] = df['cAA'].values / Kd_cAA
df['cN_Kd'] = df['cN'].values / Kd_cN
df['gamma'] = (df['cAA'].values /(df['cAA'].values + Kd_cAA))
df['nu'] = (df['cN'].values /(df['cN'].values + Kd_cN))
df['time'] = time_range
source = bokeh.models.ColumnDataSource(df)


# ############################################################################## 
# WIDGET DEFINITIONS
# ############################################################################## 
    
phiR_slider = bokeh.models.Slider(start=0, end=0.7, step=0.001, value=phi_R,
                                  title=r"ribosomal allocation")
gamma_slider = bokeh.models.Slider(start=0, end=10, step=0.01, value=gamma_max,
                                  title=r"max translational efficiency [inv. hr]")
nu_slider = bokeh.models.Slider(start=0, end=10, step=0.01, value=nu_max,
                                  title=r"max nutritional efficiency [inv. hr]")
Kd_cAA_slider = bokeh.models.Slider(start=-3, end=0, step=0.01, value=np.log10(Kd_cAA),
                                    title=r"log precursor dissociation constant")
Kd_cN_slider = bokeh.models.Slider(start=-6, end=-3, step=0.01, value=np.log10(Kd_cN),
                                    title=r"log (Monod constant [M])")
omega_slider = bokeh.models.Slider(start=16, end=25, value=np.log10(omega), 
                                    step=0.01, title=r"log (yield coefficient [AA / mol nutrient])")
cN_slider = bokeh.models.Slider(start=-6, end=0, value=np.log10(cN_0), step=0.01, 
                                    title='log (nutrient concentration [M])')

# ############################################################################## 
# CANVAS DEFINITION 
# ############################################################################## 
# Define the canvases
biomass_axis = bokeh.plotting.figure(width=600, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label='relative biomass',
                                     y_axis_type='log',
                                     title='biomass dynamics',
                                     y_range=[1, 5000]
                                     )
precursor_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$c_{AA} / K_D^{(c_{AA})}$$",
                                     title='precursor dynamics', 
                                     y_axis_type='log',
                                     y_range = [1E-4, 100]
                                     )
nutrient_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$c_N / K_D^{(c_N)}$$",
                                     title='nutrient dynamics',
                                     y_axis_type='log' ,
                                     y_range=[1E-4, 100]
                                     )
gamma_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$\gamma / \gamma_{max}$$",
                                     title='translational efficiency',
                                     y_range=[0, 1.1]
                                     )
nu_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$\nu / \nu_{max}$$",
                                     title='nutritional efficiency',
                                     y_range=[0,  1]
                                     )

# ############################################################################## 
# GLYPH DEFINITION 
# ############################################################################## 
biomass_axis.line(x='time', y='rel_M', source=source, 
                color=colors['primary_blue'], line_width=2)
precursor_axis.line(x='time', y='cAA_Kd', source=source, 
                color=colors['primary_purple'], line_width=2)
nutrient_axis.line(x='time', y='cN_Kd', source=source, 
                color=colors['primary_black'], line_width=2)
gamma_axis.line(x='time', y='gamma', source=source, line_width=2,
                color=colors['primary_green'])
nu_axis.line(x='time', y='nu', source=source, line_width=2,
                color=colors['primary_red'])

# ############################################################################## 
# CALLBACK DEFINITION
# ############################################################################## 
args = {'phiR_slider': phiR_slider,
        'gamma_slider':gamma_slider,
        'nu_slider': nu_slider,
        'Kd_cAA_slider': Kd_cAA_slider,
        'Kd_cN_slider': Kd_cN_slider,
        'cN_slider': cN_slider,
        'omega_slider': omega_slider,
        'source': source}
callback = growth.viz.load_js(['interactive_integration.js', 'functions.js'],
                        args=args)

# ############################################################################## 
# CALLBACK ASSIGNMENT
# ############################################################################## 
for s in [phiR_slider, Kd_cAA_slider, Kd_cN_slider, gamma_slider, nu_slider,
          omega_slider, cN_slider]:
          s.js_on_change('value', callback)

# ############################################################################## 
# LAYOUT DEFINITION 
# ############################################################################## 
widgets = bokeh.layouts.gridplot([[phiR_slider, Kd_cAA_slider],
                                    [gamma_slider, Kd_cN_slider],
                                    [nu_slider, omega_slider],
                                    [cN_slider]])
precursor_nutrient = bokeh.layouts.Row(precursor_axis, nutrient_axis)
efficiencies = bokeh.layouts.Row(gamma_axis, nu_axis)
layout = bokeh.layouts.gridplot([[widgets], [biomass_axis,], [precursor_nutrient], [efficiencies]])
bokeh.io.save(layout)
# %%

# %%

# 
