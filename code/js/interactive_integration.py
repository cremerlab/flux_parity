#%%
import math
import numpy as np 
import pandas as pd
import scipy.integrate
import bokeh.models 
import bokeh.plotting 
import bokeh.layouts
import bokeh.transform
import bokeh.io 
import growth.model
import growth.viz
bokeh.io.output_file('../../figures/interactive/interactive_integration.html')
colors, palette = growth.viz.bokeh_style() 
const = growth.model.load_constants()

# Set starting values for params
gamma_max = const['gamma_max']
nu_max = 4.5 
Y = const['Y']
Kd_cpc = const['Kd_cpc']
Kd_cnt = const['Kd_cnt']
phi_Rb = opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'],
                                                          nu_max,
                                                          const['Kd_cpc'],
                                                          const['phi_O'])
phi_O = const['phi_O']
phi_Mb = 1 - phi_Rb - phi_O

# Set initial conditions
OD_CONV = 1.5E17
M0 = 0.001 * OD_CONV
MR_0 = phi_Rb * M0
MP_0 = (1 - phi_Rb - phi_O) * M0
cpc_0 = 0.010
cnt_0 = 0.010

# Perform the integration and instantiate data source
time_range = np.linspace(0, 20, 4000)
params = [M0, MR_0, MP_0, cpc_0, cnt_0]
args = (gamma_max, nu_max, Y, phi_Rb, phi_Mb, Kd_cpc, Kd_cnt)
out = scipy.integrate.odeint(growth.model.self_replicator,
                            params, time_range, args=args)
df = pd.DataFrame(out, columns=['M', 'M_Rb', 'M_Mb', 'cpc', 'cnt'])
df['rel_M'] = df['M'].values / M0
df['cpc_Kd'] = df['cpc'].values / Kd_cpc
df['cnt_rel'] = df['cnt'].values / cnt_0
df['gamma'] = (df['cpc'].values /(df['cpc'].values + Kd_cpc))
df['nu'] = (df['cnt'].values /(df['cnt'].values + Kd_cnt))
df['time'] = time_range

# Generate the pie data source
angles = np.array([phi_Mb * 2 * math.pi, phi_O * 2 * math.pi, phi_Rb * 2 * math.pi])
pie_df = pd.DataFrame(np.array([angles, 
                                ['Metabolic', ' Other', 'Ribosomal'],
                                [colors['primary_purple'], colors['light_black'], 
                                colors['primary_gold']]]).T,
                        columns=['angle', 'sector', 'color'])
pie_df['start_angle'] = [0, angles[0], angles[0] + angles[1]]
pie_df['end_angle'] = [angles[0], angles[0] + angles[1], angles[0] + angles[1] + angles[2]]
pie_source = bokeh.models.ColumnDataSource(pie_df)

source = bokeh.models.ColumnDataSource(df)


# ############################################################################## 
# WIDGET DEFINITIONS
# ############################################################################## 
    
phiRb_slider = bokeh.models.Slider(start=0, end=1 - phi_O, step=0.001, value=phi_Rb,
                                  title=r"ribosomal allocation",
                                  bar_color=colors['primary_gold'])
phiO_slider = bokeh.models.Slider(start=0, end=0.75, step=0.001, value=phi_O, 
                                  title=r"allocation to other proteins",
                                  bar_color=colors['light_black'])
gamma_slider = bokeh.models.Slider(start=0, end=25, step=0.01, value=20,
                                  title=r"max translation speed [AA / s]",
                                  bar_color=colors['primary_green'])
nu_slider = bokeh.models.Slider(start=0, end=20, step=0.01, value=nu_max,
                                  title=r"max metabolic rate [inv. hr]",
                                  bar_color=colors['primary_green'])
Kd_cpc_slider = bokeh.models.Slider(start=-3, end=0, step=0.01, value=np.log10(Kd_cpc),
                                    title=u"log\u2081\u2080 precursor Michaelis-Menten constant",
                                    bar_color=colors['pale_black'])
Kd_cnt_slider = bokeh.models.Slider(start=-6, end=-3, step=0.01, value=np.log10(Kd_cnt),
                                    title=u"log\u2081\u2080 Monod constant [M]",
                                    bar_color=colors['pale_black'])
Y_slider = bokeh.models.Slider(start=16, end=25, value=np.log10(Y), 
                                    step=0.01, title=u"log\u2081\u2080 yield coefficient [(AA / mol nutrient) • vol]",
                                    bar_color=colors['pale_black'])
cnt_slider = bokeh.models.Slider(start=-6, end=0, value=np.log10(cnt_0), step=0.01, 
                                    title=u"log\u2081\u2080 nutrient concentration [M]",
                                    bar_color=colors['primary_blue'])

# ############################################################################## 
# CANVAS DEFINITION 
# ############################################################################## 
# Define the canvases

composition_axis = bokeh.plotting.figure(width=300, height=250,
                                        title='steady-state proteome composition',
                                        x_range = [-0.5, 1],
                                        y_range = [0.5, 1.5]
                                        )
composition_axis.axis.axis_label = None
composition_axis.axis.visible = False
composition_axis.grid.grid_line_color = None
composition_axis.background_fill_color = None
composition_axis.outline_line_color = None

biomass_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label='relative biomass',
                                     y_axis_type='log',
                                     title='biomass dynamics'
                                     )

precursor_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$c_{pc} / K_M^{(c_{pc})}$$",
                                     title='precursor dynamics', 
                                     )

nutrient_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"relative nutrient concentration",
                                     title='nutrient dynamics',
                                     y_range=[0, 1.1]
                                     )
gamma_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$v_{tl}/ v_{tl}^{max}$$",
                                     title='translation speed',
                                     y_range=[0, 1.1]
                                     )
nu_axis = bokeh.plotting.figure(width=300, height=300, 
                                     x_axis_label='time [hr]',
                                     y_axis_label=r"$$\nu / \nu_{max}$$",
                                     title='metabolic rate',
                                     y_range=[0,  1.1],
                                     toolbar_location=None
                                     )

# ############################################################################## 
# GLYPH DEFINITION 
# ############################################################################## 
composition_axis.wedge(x=0, y=1, radius=0.4, 
                      start_angle='start_angle',
                      end_angle='end_angle', legend_field='sector',
                      line_color='white', fill_color='color',
                      source=pie_source)

biomass_axis.line(x='time', y='rel_M', source=source, 
                color=colors['primary_black'], line_width=2)
precursor_axis.line(x='time', y='cpc_Kd', source=source, 
                color=colors['primary_blue'], line_width=2)
nutrient_axis.line(x='time', y='cnt_rel', source=source, 
                color=colors['primary_blue'], line_width=2)
gamma_axis.line(x='time', y='gamma', source=source, line_width=2,
                color=colors['primary_green'])
nu_axis.line(x='time', y='nu', source=source, line_width=2,
                color=colors['primary_green'])

# ############################################################################## 
# CALLBACK DEFINITION
# ############################################################################## 
args = {'phiRb_slider': phiRb_slider,
        'phiO_slider': phiO_slider,
        'gamma_slider':gamma_slider,
        'nu_slider': nu_slider,
        'Kd_cpc_slider': Kd_cpc_slider,
        'Kd_cnt_slider': Kd_cnt_slider,
        'cnt_slider': cnt_slider,
        'Y_slider': Y_slider,
        'source': source,
        'pie_source': pie_source}
callback = growth.viz.load_js(['interactive_integration.js', 'functions.js'],
                        args=args)

# ############################################################################## 
# CALLBACK ASSIGNMENT
# ############################################################################## 
for s in [phiRb_slider, phiO_slider, Kd_cpc_slider, Kd_cnt_slider, gamma_slider, nu_slider,
          Y_slider, cnt_slider]:
          s.js_on_change('value', callback)

# ############################################################################## 
# LAYOUT DEFINITION 
# ############################################################################## 
widgets = bokeh.layouts.gridplot([[phiRb_slider, phiO_slider],
                                    [Kd_cpc_slider, Kd_cnt_slider],
                                    [gamma_slider, nu_slider],
                                    [cnt_slider, Y_slider]])
row0 = bokeh.layouts.row(widgets, composition_axis)
row1 = bokeh.layouts.Row(biomass_axis, precursor_axis, nutrient_axis)
row2 = bokeh.layouts.Row(gamma_axis, nu_axis)

layout = bokeh.layouts.gridplot([[row0], [row1], [row2]])
bokeh.io.save(layout)
# %%

# %%

# 
