#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import scipy.integrate
import growth.viz
import seaborn as sns
import growth.model
import tqdm
const = growth.model.load_constants()
colors, palette = growth.viz.matplotlib_style()
data = pd.read_csv('../../data/Scott2010_chlor_inhibition_minimal.csv')
data['mass_fraction'] = data['RNA_protein_ratio'].values * 0.4558
#%%
gamma_max = const['gamma_max']
Kd_cpc = const['Kd_cpc']
tau = const['tau']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_max = const['kappa_max']
phi_O = 0.75 
c_ab = np.linspace(0, 15, 100) * 1E-6 # in M
Kd_cab = 5E-10  # in M


# Define the desired growth rates
target_lam = {'M63 + glycerol': 0.4,
              'M63 + glucose': 0.57,
              'cAA + glucose': 1,
              'cAA + glycerol':0.71}

target_phiRb = {'M63 + glycerol': 0.177 * .4558,
               'M63 + glucose': 0.230 * 0.4558,
               'cAA + glucose': 0.287 * 0.4558,
               'cAA + glycerol': 0.224 * 0.4558}

def compute_nu(gamma_max, Kd, phiRb, lam):
    return (lam / (1 - phiRb - phi_O)) * (((lam * Kd)/(gamma_max * phiRb - lam)) + 1)
target_nu = {k: compute_nu(gamma_max, Kd_cpc, target_phiRb[k], v) for k, v in target_lam.items()}

nu_max = target_nu.values
#%%
# Perform the chlor integration
df = pd.DataFrame([])
dt = 0.001
time = np.arange(0, 18, dt)

for k, v in tqdm.tqdm(target_nu.items()):
    # Instantiate
    init_nu = 2.3 
    # opt_phiRb = growth.model.phiRb_optimal_allocation(gamma_max, init_nu, const['Kd_cpc'], phi_O)
    opt_phiRb = 0.2
    M0 = 1E9
    Mrb = opt_phiRb * M0
    Mmb = (1 - opt_phiRb - phi_O) * M0
    TAA = 0.0002
    TAA_star = 0.00002

    # Pack
    # args = (gamma_max, v, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O)
    # params = [M0, Mrb, Mmb, TAA, TAA_star]
    # out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp, params, time, args=args)
    # out = out[-1]
    # opt_phiRb = out[1] / out[0]
    # TAA = out[-2]
    # TAA_star = out[-1]
    params = [M0, opt_phiRb * M0, (1 - opt_phiRb - phi_O) * M0, TAA, TAA_star]

    for j, c in enumerate(c_ab):
        args = (gamma_max, v * 1.12, tau, Kd_TAA, Kd_TAA_star, kappa_max, phi_O, c, Kd_cab)
        _out = scipy.integrate.odeint(growth.model.self_replicator_ppGpp_chlor,
        params, time, args=args) 
        out = _out[-1]
        df = df.append({'M':out[0],
                    'Mrb': out[1],
                    'Mmb': out[2],
                    'TAA': out[-2],
                    'TAA_star': out[-1],
                    'lam': np.log(_out[-1][0] / _out[-2][0])/dt,
                    'chlor_conc_uM': c,
                    'nu_max': v
                    }, ignore_index=True)
#%%
# Compute properties
df['balance'] = df['TAA_star'].values / df['TAA'].values
df['phi_Rb'] = df['balance'].values / (df['balance'].values + tau)
df['MRb_M'] = df['Mrb'].values / df['M'].values

# %%
fig, ax = plt.subplots(1, 1)
ax.set_xlabel('growth rate [per hr]')
ax.set_ylabel('$M_{Rb} / M$')

counter = 0
for g, d in data.groupby(['medium']):
    _d = df[df['nu_max']==target_nu[g]]
    ax.plot(d['growth_rate_hr'], d['mass_fraction'], 'o', color=palette[counter], label=g)
    ax.plot(_d['lam'], _d['MRb_M'], '--', color=palette[counter], label='__nolegend__')
    counter += 1


ax.legend()
plt.savefig('./Scott2010_chloramphenicol.pdf', bbox_inches='tight')
# %%

# %%
