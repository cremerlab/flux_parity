#%%
import numpy as np 
import pandas as pd 
import bokeh.io 
import bokeh.plotting 
import bokeh.layouts
import bokeh.models
import growth.model
import growth.viz 
colors, palette = growth.viz.bokeh_style()
bokeh.io.output_file('./interactive_steadystate.html')

# ############################################################################## 
# CONSTANT DEFINITIONS
# ############################################################################## 

phiR_range = np.linspace(0.0001, 0.9999, 500)
gamma_max = 9.65
nu_max = 10 
Kd_cAA = 0.025

# ############################################################################## 
# INITIALIZATION    
# ############################################################################## 
growth_rate = growth.model.steady_state_mu(gamma_max, phiR_range, nu_max, Kd_cAA)
cAA = growth.model.steady_state_cAA(gamma_max, phiR_range, nu_max, Kd_cAA)
gamma = growth.model.steady_state_gamma(gamma_max, phiR_range, nu_max, Kd_cAA)
df = pd.DataFrame(np.array([phiR_range, growth_rate, cAA/Kd_cAA, gamma/gamma_max]).T,
                   columns=['phiR', 'mu', 'cAA', 'gamma'])
source = bokeh.models.ColumnDataSource(df)

# ############################################################################## 
# WIDGET DEFINITIONS
# ############################################################################## 

gamma_slider = bokeh.models.Slider(start=0, end=15, step=0.001, value=gamma_max,
                    title='maximum translational efficiency [inv. hr]')
nu_slider = bokeh.models.Slider(start=0, end=15, step=0.001, value=nu_max,
                    title='maximal nutritional efficiency [inv. hr]')
Kd_cAA_slider = bokeh.models.Slider(start=-4, end=0, step=0.001, value=np.log10(Kd_cAA),
                    title='log10 effective dissociation constant')

# ############################################################################## 
# CANVAS DEFINITIONS
# ############################################################################## 

growth_axis = bokeh.plotting.figure(width=300, height=300, 
                                    x_axis_label='ribosomal allocation factor',
                                    y_axis_label='growth rate [per hr]',
                                    title='steady state growth rate')
precursor_axis = bokeh.plotting.figure(width=300, height=300,
                                    x_axis_label='ribosomal allocation factor',
                                    y_axis_label=r"$$c_{AA} / K_D^{(c_{AA})}$$",
                                    title='steady state precursor concentration',
                                    y_axis_type='log')
gamma_axis = bokeh.plotting.figure(width=300, height=300,
                                    x_axis_label='ribosomal allocation factor',
                                    y_axis_label=r"$$\gamma / \gamma_{max}$$",
                                    title='steady state translational efficiency')

# ############################################################################## 
# GLYPH DEFINITIONS
# ############################################################################## 
growth_axis.line(x='phiR', y='mu', source=source, 
                color=colors['primary_blue'], line_width=2)
precursor_axis.line(x='phiR', y='cAA', source=source, 
                color=colors['primary_purple'], line_width=2)
gamma_axis.line(x='phiR', y='gamma', source=source, 
                color=colors['primary_green'], line_width=2)

# ############################################################################## 
# CALLBACK DEFINITIONS
# ############################################################################## 
args = {'gamma_slider': gamma_slider,
        'nu_slider': nu_slider,
        'Kd_cAA_slider': Kd_cAA_slider,
        'source': source}
callback = growth.viz.load_js(['./interactive_steadystate.js', './functions.js'],
                        args=args)
for s in [gamma_slider, nu_slider, Kd_cAA_slider]:
    s.js_on_change('value', callback)

# ############################################################################## 
# LAYOUT DEFINITION
# ############################################################################## 
sliders = bokeh.layouts.column(gamma_slider, nu_slider, Kd_cAA_slider)
layout = bokeh.layouts.gridplot([[sliders, growth_axis],
                                 [precursor_axis, gamma_axis]])
bokeh.io.save(layout)
# %%
