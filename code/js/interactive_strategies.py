#%%
import numpy as np 
import pandas as pd 
import bokeh.io 
import bokeh.plotting 
import bokeh.models
import growth.viz 
import growth.model
import seaborn as sns 
colors, palette = growth.viz.bokeh_style()
const = growth.model.load_constants()
bokeh.io.output_file('../../figures/interactive/interactive_strategies.html')

# ############################################################################## 
# PARAMETER DEFINITIONS
# ############################################################################## 
phi_O = 0.55
phiRb_range = np.arange(0.001, 1 - phi_O - 0.001, 0.001)
nu_max= np.arange(0.001, 20, 0.1)

const_phiRb = 0.25
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']

# Set up the initial scenarios
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O)
const_phiRb = const_phiRb * np.ones_like(nu_max)
const_gamma = growth.model.steady_state_gamma(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O)
const_lam = growth.model.steady_state_growth_rate(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O)
trans_phiRb = growth.model.phiRb_constant_translation(gamma_max, nu_max, 10, Kd_cpc, phi_O)
trans_gamma = growth.model.steady_state_gamma(gamma_max, trans_phiRb, nu_max, Kd_cpc, phi_O)
trans_lam = growth.model.steady_state_growth_rate(gamma_max, trans_phiRb, nu_max, Kd_cpc, phi_O)


# Set up the source
source = bokeh.models.ColumnDataSource({
                        'nu_max': [nu_max, nu_max, nu_max],
                        'phiRb':  [const_phiRb, trans_phiRb, opt_phiRb],
                        'gamma': list(np.array([const_gamma, trans_gamma, opt_gamma]) * 7459 / 3600),
                        'lam': [const_lam, trans_lam, opt_lam],
                        'color': [colors['primary_black'], colors['primary_green'], colors['primary_blue']],
                        'label': ['scenario I: constant allocation', 'scenario II: constant translation rate', 'scenario III: optimal allocation'],
                        'filler_xs': [[], [], []],
                        'filler_ys': [[], [], []]
})


# ############################################################################## 
# WIDGET DEFINITIONS
# ############################################################################## 
phiO_slider = bokeh.models.Slider(start=0, end=0.75, step=0.001, value=phi_O,
                    title='allocation to other proteins')
gamma_slider = bokeh.models.Slider(start=0.001, end=25, step=0.001, value=gamma_max * 7459 / 3600,
                    title='maximum translation speed [AA / s]')
sc2_slider = bokeh.models.Slider(start=0.001, end=0.999, step=0.001, value=0.9,
                    title='scenario II: target translation speed (of maximum)',
                    default_size=350,
                    bar_color=colors['primary_green'])
Kd_cpc_slider = bokeh.models.Slider(start=-4, end=-0.0001, step=0.001, value=np.log10(Kd_cpc),
                    title='log\u2081\u2080 precursor Michaelis-Menten constant',
                    default_size=350)
phiRb_slider = bokeh.models.Slider(start=0.001, end=1 - phi_O, step=0.001,
                    value = 0.25,
                    title='scenario I: constant ribosomal allocation parameter',
                    bar_color=colors['primary_black'],
                    default_size=350)

# ############################################################################## 
# AXES DEFINITION
# ############################################################################## 
phiRb_axis = bokeh.plotting.figure(width=300, height=300,
                                    x_axis_label='maximum metabolic rate ν',
                                    y_axis_label=r"$$\phi_{Rb}$$",
                                    title='steady-state ribosomal content',
                                    y_range = [0, 0.5])
                                 
gamma_axis = bokeh.plotting.figure(width=300, height=300,
                                    x_axis_label='maximum metabolic rate ν',
                                    y_axis_label="γ [inv. hr]",
                                    title='steady-state translation rate',
                                    y_range = [0, 28])
lam_axis = bokeh.plotting.figure(width=300, height=300,
                                    x_axis_label='maximum metabolic rate ν',
                                    y_axis_label="λ [inv. hr]",
                                    title='steady-state growth rate',
                                    y_range=[0, 2.5])

legend_axis = bokeh.plotting.figure(width=370, height=120, tools=[])
legend_axis.axis.axis_label = None
legend_axis.axis.visible = False
legend_axis.grid.grid_line_color = None
legend_axis.background_fill_color = None
legend_axis.outline_line_color = None


# ##############################################################################
# GLYPH DEFINITIONS
# ##############################################################################
phiRb_axis.multi_line(xs='nu_max', ys='phiRb', line_color='color', 
                     line_width=2, source=source)
gamma_axis.multi_line(xs='nu_max', ys='gamma', line_color='color',
                     line_width=2, source=source)
lam_axis.multi_line(xs='nu_max', ys='lam', line_color='color', 
                     line_width=2, source=source)

legend_axis.multi_line(xs='filler_xs', ys='filler_ys', line_width=2.5,
                        line_color='color', legend_field='label' ,
                        source=source)
# ##############################################################################
# CALLBACK DEFINITION 
# ##############################################################################
args = {'gamma_slider': gamma_slider,
        'sc2_gamma_slider': sc2_slider,
        'Kd_cpc_slider': Kd_cpc_slider,
        'phiO_slider': phiO_slider,
        'phiRb_slider': phiRb_slider,
        'source': source} 
callback = growth.viz.load_js(['./interactive_strategies.js', './functions.js'],
                        args=args)
for s in [gamma_slider, Kd_cpc_slider, phiO_slider,  phiRb_slider, sc2_slider]:
    s.js_on_change('value', callback)

# ##############################################################################
# LAYOUT AND SAVING
# ##############################################################################
sliders1 = bokeh.layouts.Column(gamma_slider,  Kd_cpc_slider)
sliders2 = bokeh.layouts.Column(phiO_slider, phiRb_slider, sc2_slider)
row1 = bokeh.layouts.Row(sliders1, sliders2, legend_axis)
row2 = bokeh.layouts.Row(phiRb_axis, gamma_axis, lam_axis)
layout = bokeh.layouts.Column(row1, row2)
bokeh.io.save(layout)



# %%
#