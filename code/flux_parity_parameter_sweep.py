import numpy as np 
import pandas as pd 
import growth.integrate
import tqdm
const = growth.model.load_constants()

nu_max = [0.1, 0.5, 4.5, 10, 15]
gamma_max = const['gamma_max']
phi_O = const['phi_O']
Kd_TAA = const['Kd_TAA']
Kd_TAA_star = const['Kd_TAA_star']
kappa_range = np.logspace(-6, 0, 100)
tau_range = np.logspace(-4, 2, 100)

df = pd.DataFrame([])
for i, tau in enumerate(tqdm.tqdm(tau_range, desc='Iterating through tau')):
    for j, kappa in enumerate(tqdm.tqdm(kappa_range, desc='Iterating through kappa')):
        for k, nu in enumerate(tqdm.tqdm(nu_max, desc="Iterationg through nu")):
            _args = {'gamma_max':gamma_max,
                    'nu_max':nu,
                    'Kd_TAA':Kd_TAA,
                    'Kd_TAA_star':Kd_TAA_star,
                    'kappa_max':kappa,
                    'phi_O':phi_O,
                    'tau': tau}
            out = growth.integrate.equilibrate_FPM(_args, max_iter=20)
            _df = pd.DataFrame([out[-2:]], columns=['TAA', 'TAA_star'])
            _df['gamma'] = gamma_max * out[-1] / (out[-1] + Kd_TAA_star)
            _df['balance'] = out[-1] / out[-2]
            _df['phiRb'] = (1 - const['phi_O']) * _df['balance'] / (_df['balance'] + tau)
            _df['tau'] = tau
            _df['kappa_max'] = kappa
            _df['growth_rate'] = _df['gamma'] * _df['phiRb']
            _df['nu_max'] = nu
            df = pd.concat([df, _df], ignore_index=True)  

df.to_csv('../data/flux_parity_parameter_sweep.csv', index=False)