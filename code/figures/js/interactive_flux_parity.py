#%%
import numpy as np 
import pandas as pd 
import scipy.integrate
import bokeh.io
import bokeh.plotting
import bokeh.models
import growth.model
import growth.viz
import scipy.integrate
const = growth.model.load_constants()
colors, palette = growth.viz.bokeh_style()
bokeh.io.output_file('interactive_flux_parity.html')

# Define model constants
gamma_max = const['gamma_max']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
tau = const['tau']
nu_max = 4
phi_O = 0.55

# Set the initial conditions
phiRb_0 = 0.25
M0 = 1
MRb = phiRb_0 * M0
MMb = (1 - phiRb_0 - phi_O) * M0
TAA = 1E-5
TAA_star = 1E-6
dt = 0.001
time_range = np.arange(0, 10, dt)
# Integrate
args = (gamma_max, nu_max, tau, Kd_TAA, Kd_TAA_star, kappa_max, tau)
params = [M0, MRb, MMb, TAA, TAA_star]
out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, 
                             params, 
                             time_range,
                             args=args)

# Set up the data source
data = pd.DataFrame(out, columns=['M', 'MRb', 'MMb', 'TAA', 'TAA_star'])
data['ratio'] = data['TAA_star'].values / data['TAA'].values
data['MRb_M'] = data['MRb'].values / data['M'].values
data['MMb_M'] = data['MMb'].values / data['M'].values
data['phiRb'] = data['ratio'].values / (data['ratio'].values + tau)
data['tot_tRNA'] = data['TAA'].values + data['TAA_star'].values
gr = list(np.log(data['M'].values[1:] / data['M'].values[:-1]) / dt)
gr.append(gr[-1])
source = bokeh.models.ColumnDataSource(data)


# Set up the axes
phiRb_range = np.linspace(0, 1 - phi_O -0.001, 100)
simple_lam = growth.model.steady_state_growth_rate(gamma_max, phiRb_range, nu_max, const['Kd_cpc'], phi_O) 

