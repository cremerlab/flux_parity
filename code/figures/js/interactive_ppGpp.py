#%%
import numpy as np 
import pandas as pd 
import bokeh.io 
import bokeh.model 
import bokeh.plotting
import bokeh.layouts
import growth.viz 
import growth.model
import scipy.integrate
import tqdm
bokeh.io.output_file('./interactive_ppGpp.html')


# Define constants
colors, palette = growth.viz.bokeh_style()
const = growth.model.load_constants()
mapper = growth.viz.load_markercolors()
const = growth.model.load_constants()
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
nu_max = np.linspace(0.001, 8, 100)
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
tau = const['tau']
kappa_max = const['kappa_max']

# Load the data 
mass_frac = pd.read_csv('../../../data/ribosomal_mass_fractions.csv')
mass_frac = mass_frac[(mass_frac['organism']=='Escherichia coli') & 
                      (mass_frac['source'] != 'Wu et al., 2021')]
elong_data = pd.read_csv('../../../data/peptide_elongation_rates.csv')
elong_data = elong_data[(elong_data['organism']=='Escherichia coli') & 
                       (elong_data['source'] != 'Wu et al., 2021')]
tRNA_data = pd.read_csv('../../../data/tRNA_abundances.csv')

# Assign markers and colors
mass_frac['color'] = [mapper[k]['c'] for k in mass_frac['source'].values]
mass_frac['marker'] = [mapper[k]['m_bokeh'] for k in mass_frac['source'].values]
elong_data['color'] = [mapper[k]['c'] for k in elong_data['source'].values]
elong_data['marker'] = [mapper[k]['m_bokeh'] for k in elong_data['source'].values]
tRNA_data['color'] = [mapper[k]['c'] for k in tRNA_data['source'].values]
tRNA_data['marker'] = [mapper[k]['m_bokeh'] for k in tRNA_data['source'].values]

# Assemble data CDS
mass_frac = bokeh.models.ColumnDataSource(mass_frac)
elong_data = bokeh.models.ColumnDataSource(elong_data)
tRNA_data = bokeh.models.ColumnDataSource(tRNA_data)

# Compute the theory data
phi_O = 0.55
opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu_max, Kd_cpc, phi_O)
opt_lam = growth.model.steady_state_growth_rate(gamma_max,  opt_phiRb, nu_max, Kd_cpc, phi_O)
opt_gamma = growth.model.steady_state_gamma(gamma_max, opt_phiRb,  nu_max, Kd_cpc, phi_O) * 7459 / 3600

# Numerically compute the optimal scenario
dt = 0.0001
time_range = np.arange(0, 200, dt)
ss_df = pd.DataFrame([])
total_tRNA = 0.0004
T_AA = total_tRNA / 2
T_AA_star = total_tRNA / 2
for i, nu in enumerate(tqdm.tqdm(nu_max)):
    # Set the intitial state
    _opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max,  nu, Kd_cpc) 
    M0 = 1E9
    phi_Mb = 1 -  _opt_phiRb
    M_Rb = _opt_phiRb * M0
    M_Mb = phi_Mb * M0
    params = [M0, M_Rb, M_Mb, T_AA, T_AA_star]
    args = (gamma_max, nu, tau, Kd_TAA, Kd_TAA_star, kappa_max, 0.25)

    # Integrate
    out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp,
                                params, time_range,  args=args)
    # Compute the final props
    _out = out[-1]
    ratio = _out[-1] / _out[-2]
    tRNA_abund = _out[-2] + _out[-1]
    biomass = _out[0]
    ss_df = ss_df.append({'phi_Rb': _out[1] / _out[0],
                          'lam': np.log(_out[0] / out[-2][0]) / (time_range[-1] - time_range[-2]),
                          'gamma': gamma_max * _out[-1]/ (_out[-1]+ Kd_TAA_star) * 7459 / 3600,
                          'balance': ratio,
                          'tot_tRNA_abundance': tRNA_abund,
                          'biomass': biomass,
                          'tRNA_per_ribosome': (tRNA_abund * biomass) / (_out[1]/7459), 
                          'nu_max': nu},
                          ignore_index=True)


# Assemble the theory CDS
theory = bokeh.models.ColumnDataSource({ 'nu_max': [nu_max, nu_max],
                                        'lam': [opt_lam, ss_df['lam'].values],
                                        'phiRb': [opt_phiRb, ss_df['phi_Rb'].values],
                                        'gamma': [opt_gamma, ss_df['gamma'].values],
                                        'tRNA_ribo': [[], ss_df['tRNA_per_ribosome'].values],
                                        'linestyle': ['solid', 'dashed'],
                                        'color': [colors['primary_blue'], colors['primary_red']]
                                        })
# ##############################################################################
# WIDGET DEFINITION
# ##############################################################################
Kd_TAA_slider = bokeh.models.Slider(start=-8, end=-3, step=0.001, value=np.log10(Kd_TAA),
                                    title='log\u2081\u2080 ucharged tRNA dissociation constant')
Kd_TAA_star_slider = bokeh.models.Slider(start=-8, end=-3, step=0.001, value=np.log10(Kd_TAA_star),
                                    title='log\u2081\u2080 charged tRNA dissociation constant')
tau_slider = bokeh.models.Slider(start=0.1, end=10, step=0.001, value=tau,
                                    title='threshold charged-tRNA/uncharged-tRNA balance')
kappa_slider = bokeh.models.Slider(start=-5, end=0, step=0.001, value=np.log10(kappa_max),
                                    title='log\u2081\u2080 tRNA synthesis rate')

# ##############################################################################
# CANVAS DEFINITION
# ##############################################################################
allocation_ax = bokeh.plotting.figure(width=350, height=350, 
                                 y_axis_label='allocation towards ribosomes',
                                 x_axis_label='growth rate [inv. hr.]')

elongation_ax = bokeh.plotting.figure(width=350, height=350, 
                                 y_axis_label='translation speed [AA / s]',
                                 x_axis_label='growth rate [inv. hr.]')

tRNA_ax = bokeh.plotting.figure(width=350, height=350, 
                                 y_axis_label='tRNA per ribosome',
                                 x_axis_label='growth rate [inv. hr.]',
                                 y_range=[0, 20])

# ##############################################################################
# CANVAS POPULATION
# ##############################################################################
allocation_ax.scatter(x='growth_rate_hr',  y='mass_fraction', marker='marker',
                        color='color', source=mass_frac, size=10, line_color='black',
                        alpha=0.5, name='data')
allocation_ax.multi_line(xs='lam', ys='phiRb', line_color='color', source=theory,
                    line_width=2, line_dash='linestyle')
elongation_ax.scatter(x='growth_rate_hr',  y='elongation_rate_aa_s', marker='marker',
                        color='color', source=elong_data, size=10, line_color='black',
                        alpha=0.5, name='data')
elongation_ax.multi_line(xs='lam', ys='gamma', line_color='color', source=theory,
                    line_width=2, line_dash='linestyle')

tRNA_ax.scatter(x='growth_rate_hr',  y='tRNA_per_ribosome', marker='marker',
                        color='color', source=tRNA_data, size=10, line_color='black',
                        alpha=0.5, name='data')
tRNA_ax.multi_line(xs='lam', ys='tRNA_ribo', line_color='color', source=theory,
                    line_width=2, line_dash='linestyle')

# ##############################################################################
# CALLBACK DEFINITION
# ##############################################################################
cb = growth.viz.load_js(['./interactive_ppGpp.js', './functions.js'], 
                args={'Kd_TAA_slider': Kd_TAA_slider,
                      'Kd_TAA_star_slider': Kd_TAA_star_slider,
                      'kappa_slider': kappa_slider,
                      'tau_slider': tau_slider,
                      'source': theory})
for s in [Kd_TAA_slider, Kd_TAA_star_slider, kappa_slider, tau_slider]:
    s.js_on_change('value', cb)
# ##############################################################################
# LAYOUT AND SAVING
# ##############################################################################
widgets = bokeh.layouts.Column(Kd_TAA_star_slider, Kd_TAA_slider, kappa_slider, tau_slider)
row1 = bokeh.layouts.Row(widgets, allocation_ax)
row2 = bokeh.layouts.Row(elongation_ax, tRNA_ax)
layout = bokeh.layouts.Column(row1, row2)
bokeh.io.save(layout)
# %%
