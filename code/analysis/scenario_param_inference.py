#%%
import numpy as np 
import pandas as pd 
import bebi103.stan
import cmdstanpy 
import arviz as az 

# Load the datasets 
frac_data = pd.read_csv('../../data/mass_fraction_compiled.csv')
frac_data = frac_data[frac_data['growth_rate_hr'] >= 0.5]
elong_data = pd.read_csv('../../data/dai2016_elongation_rate.csv')
elong_data = elong_data[elong_data['growth_rate_hr'] >= 0.5]

# Merge 
elong_data['source'] = 'Dai et al., 2016'
elong_data['category'] = 'elongation rate measurement'
frac_data['category'] = 'mass fraction measurement'
frac_data = pd.concat([frac_data, elong_data[['growth_rate_hr', 
                                              'source', 
                                              'mass_fraction',
                                              'category']]],
           sort = False)
dai_idx = np.where(frac_data['category']=='elongation rate measurement')[0] + 1
frac_data.to_csv('../../data/collated_mass_fraction_measurements.csv', index=False)

#%%
# Load and compile the model 
opt_model = cmdstanpy.CmdStanModel(stan_file='../stan/optimal_allocation_inference.stan')

# Define the data dictionary
data_dict = {'N_frac':len(frac_data),
             'N_elong': len(elong_data),
             'elong_idx': dai_idx.astype(int),
             'growth_rate': frac_data['growth_rate_hr'].values.astype(float),
             'mass_frac': frac_data['mass_fraction'].values.astype(float), 
             'elong_rate':elong_data['elongation_rate_aa_s'].values.astype(float),
             'gamma_max_mu': 9.65,
             'gamma_max_sigma': 1,
             'phiO': 0.5}


samples = opt_model.sample(data=data_dict, adapt_delta=0.99)
samples = az.from_cmdstanpy(samples)
bebi103.stan.check_all_diagnostics(samples)
samples = samples.posterior.to_dataframe().reset_index()

# %%
#  Generate a mapper of meausured growth rate to nutritional capacity
nu_mapper = {k:v for k, v in enumerate(frac_data['growth_rate_hr'].values)}
source_mapper = {k:v for k, v in enumerate(frac_data['source'].values)}
nu_map = [nu_mapper[k] for k in samples['nu_max_dim_0'].values]
source_map = [source_mapper[k] for k in samples['nu_max_dim_0'].values]

#%%

# Save the inferred nu_max and the other parameter separately
nu_samp_dfs  = []
nu_summ_df = pd.DataFrame([])
for g, d in samples.groupby(['nu_max_dim_0']):
    _df = d['nu_max']
    _df['measured_growth_rate_hr'] = nu_mapper[g]
    _df['source'] = source_map[g]
    nu_samp_dfs.append(_df)

    # Compute the summary
    nu_mean = d['nu_max'].mean()
    nu_median = d['nu_max'].median() 
    nu_std = d['nu_max'].std()
    nu_sem = d['nu_max'].sem()
    nu_95_upper, nu_95_lower = np.percentile(d['nu_max'].values, (97.5, 2.5))
    nu_summ_df = nu_summ_df.append({
                            'measured_growth_rate_hr':nu_mapper[g],
                            'source': source_mapper[g],
                            'nu_mean':nu_mean,
                            'nu_median': nu_median,
                            'nu_sem':nu_sem,
                            'nu_std':nu_std,
                            'nu_95th_lower':nu_95_lower,
                            'nu_95th_upper':nu_95_upper}, ignore_index=True)
    
nu_samp_df = pd.concat(nu_samp_dfs, sort=False)

#%% Save the model parameters

param_summary = pd.DataFrame([])
samps = samples[['gamma_max', 'nu_max', 'Kd', 'mass_frac_sigma', 'elong_sigma']]
samps.drop_duplicates(inplace=True)
param_samples = samps.melt(var_name='parameter')
for g, d in param_samples.groupby(['parameter']):
    mean_param = d['value'].mean()
    median_param = d['value'].median()
    sem_param = d['value'].sem()
    std_param = d['value'].std()
    param_95_upper, param_95_lower = np.percentile(d['value'], (97.5, 2.5))
    param_summary = param_summary.append({'parameter': g,
                                          'param_mean':mean_param,
                                          'param_median':median_param,
                                          'param_sem':sem_param,
                                          'param_std':std_param,
                                          'param_95th_upper':param_95_upper,
                                          'param_95th_lower':param_95_lower},
                                          ignore_index=True)

#%% Save to disk
nu_samp_df.to_csv('../../data/nutritional_efficiency_optimal_allocation_MCMC_samples.csv', 
                 index=False)
nu_summ_df.to_csv('../../data/nutritional_efficiency_optimal_allocation_MCMC_summary.csv',
                 index=False)
param_samples.to_csv('../../data/optimal_allocation_MCMC_samples.csv', index=False)
param_summary.to_csv('../../data/optimal_allocation_MCMC_summary.csv', index=False)

# %%
