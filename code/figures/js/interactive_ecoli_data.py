#%%
import numpy as np 
import pandas as pd 
import bokeh.plotting 
import bokeh.io 
import bokeh.models 
import growth.model
import growth.viz
const = growth.model.load_constants()
colors, palette = growth.viz.bokeh_style()
mapper = growth.viz.load_markercolors()
bokeh.io.output_file('./interactive_ecoli_data.html')

# Define constants
gamma_max = const['gamma_max']
phi_O = 0.25
Kd_cpc = const['Kd_cpc']
nu_max= np.arange(0.001, 10, 0.001)
const_phiRb = 0.15


# Load the mass_frac 
mass_frac = pd.read_csv('../../../data/ribosomal_mass_fractions.csv')
mass_frac  = mass_frac[mass_frac['organism']=='Escherichia coli'] 
elong = pd.read_csv('../../../data/peptide_elongation_rates.csv')
elong = elong[elong['organism']=='Escherichia coli']

# Add markers and colors to maintain consistency.
markers = [mapper[g]['m_bokeh'] for g in mass_frac['source'].values]
_colors = [mapper[g]['c'] for g in mass_frac['source'].values]
mass_frac['marker'] = markers
mass_frac['color'] = _colors

markers = [mapper[g]['m_bokeh'] for g in elong['source'].values]
_colors = [mapper[g]['c'] for g in elong['source'].values]
elong['marker'] = markers
elong['color'] = _colors

mass_frac = bokeh.models.ColumnDataSource(mass_frac)
elong = bokeh.models.ColumnDataSource(elong)



# Set up the initial scenarios
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O) * 7459 / 3600
opt_lam = growth.model.steady_state_growth_rate(gamma_max, opt_phiRb, nu_max, Kd_cpc, phi_O)
const_phiRb = const_phiRb * np.ones_like(nu_max)
const_gamma = growth.model.steady_state_gamma(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O) * 7459 / 3600
const_lam = growth.model.steady_state_growth_rate(gamma_max, const_phiRb, nu_max, Kd_cpc, phi_O)
trans_phiRb = growth.model.phiRb_constant_translation(gamma_max, nu_max, phi_O)
trans_gamma = growth.model.steady_state_gamma(gamma_max, trans_phiRb, nu_max, Kd_cpc, phi_O) * 7459 / 3600
trans_lam = growth.model.steady_state_growth_rate(gamma_max, trans_phiRb, nu_max, Kd_cpc, phi_O)

source = bokeh.models.ColumnDataSource({'phiRb': [const_phiRb, trans_phiRb, opt_phiRb],
                                        'gamma': [const_gamma, trans_gamma, opt_gamma],
                                        'lam': [const_lam, trans_lam, opt_lam],
                                        'color': [colors['primary_black'], 
                                                  colors['primary_green'],
                                                  colors['primary_blue']],
                                        'label': ['scenario I: constant allocation',
                                                  'scenario II: constant translation rate',
                                                  'scenario III: optimal allocation'],
                                        'filler_xs': [[], [], []],
                                        'filler_ys': [[], [], []]})


# ############################################################################## 
# WIDGET DEFINITIONS
# ############################################################################## 
phiO_slider = bokeh.models.Slider(start=0, end=0.5, step=0.001, value=phi_O,
                    title='allocation to other proteins')
gamma_slider = bokeh.models.Slider(start=0.001, end=10, step=0.001, value=gamma_max,
                    title='maximum translation rate [inv. hr]')
Kd_cpc_slider = bokeh.models.Slider(start=-4, end=-0.0001, step=0.001, value=np.log10(Kd_cpc),
                    title='log\u2081\u2080 precursor dissociation constant')
phiRb_slider = bokeh.models.Slider(start=0.001, end=0.45, step=0.001,
                    value = 0.15,
                    title='scenario I: constant ribosomal allocation parameter',
                    bar_color=colors['primary_black'])

# ############################################################################## 
# CANVAS DEFINITION
# ############################################################################## 
mass_frac_tooltips = [('source', '@source'),
                      ('ribosomal allocation',  '@mass_fraction{0.2f}'),
                      ('growth rate\n[inv. hr.]', '@growth_rate_hr{0.2f}'),
                      ('method', '@method')]
elong_tooltips = [('source', '@source'),
                      ('translation rate [AA/s]',  '@elongation_rate_aa_s{0.2f}'),
                      ('growth rate\n[inv. hr.]', '@growth_rate_hr{0.2f}')]

mass_hover = bokeh.models.HoverTool(names=['data'], tooltips=mass_frac_tooltips)
elong_hover = bokeh.models.HoverTool(names=['data'], tooltips=elong_tooltips)

allocation_axis = bokeh.plotting.figure(width=450, height=400,
                                        x_axis_label='growth rate λ [inv. hr]',
                                        y_axis_label = 'ribosomal allocation',
                                        y_range=[0, 0.35],
                                        x_range=[0, 2],
                                        tools = [mass_hover, 'pan', 
                                                'wheel_zoom', 'box_zoom']
                                        )

elongation_axis = bokeh.plotting.figure(width=450, height=400,
                                        y_axis_label='translation rate γ [AA / s]',
                                        x_axis_label = 'growth rate λ [inv. hr]',
                                        y_range=[5, 20],
                                        x_range = [0, 2],
                                        tools = [elong_hover, 'pan', 
                                                'wheel_zoom', 'box_zoom']
                                        )

legend_axis = bokeh.plotting.figure(width=370, height=120, tools=[])
legend_axis.axis.axis_label = None
legend_axis.axis.visible = False
legend_axis.grid.grid_line_color = None
legend_axis.background_fill_color = None
legend_axis.outline_line_color = None

# ############################################################################## 
# GLYPH DEFINITION
# ############################################################################## 
allocation_axis.scatter(x='growth_rate_hr',  y='mass_fraction', marker='marker',
                        color='color', source=mass_frac, size=10, line_color='black',
                        alpha=0.75, name='data')
elongation_axis.scatter(x='growth_rate_hr', y='elongation_rate_aa_s', marker='marker',
                       color='color', source=elong, size=10, line_color='black',
                       alpha=0.75, name='data')

allocation_axis.multi_line(xs='lam', ys='phiRb', color='color', line_width=2,
                            source=source)
elongation_axis.multi_line(xs='lam', ys='gamma', color='color', line_width=2,
                            source=source)
legend_axis.multi_line(xs='filler_xs', ys='filler_ys', line_width=2.5,
                        line_color='color', legend_field='label' ,
                        source=source)
 ##############################################################################
# CALLBACK DEFINITION 
# ##############################################################################
args = {'gamma_slider': gamma_slider,
        'Kd_cpc_slider': Kd_cpc_slider,
        'phiO_slider': phiO_slider,
        'phiRb_slider': phiRb_slider,
        'source': source,
        'nu_max': nu_max} 

callback = growth.viz.load_js(['./interactive_ecoli_data.js', './functions.js'],
                        args=args)
for s in [gamma_slider, Kd_cpc_slider, phiO_slider,  phiRb_slider]:
    s.js_on_change('value', callback)


# ############################################################################## 
# LAYOUT
# ##############################################################################
col1 = bokeh.layouts.Column(gamma_slider, phiO_slider)
col2 = bokeh.layouts.Column(Kd_cpc_slider, phiRb_slider)
sliders = bokeh.layouts.Row(col1, col2, legend_axis)
row1 = bokeh.layouts.Row(allocation_axis, elongation_axis)
layout = bokeh.layouts.Column(sliders, row1)
bokeh.io.save(layout)
# %%
