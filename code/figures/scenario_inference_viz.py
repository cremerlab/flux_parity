#%%
import numpy as np 
import pandas as pd 
import growth.model
import growth.viz
import altair as alt
colors, palette = growth.viz.altair_style()
alt.data_transformers.disable_max_rows()

# Load the experimental data. 
data = pd.read_csv('../../data/collated_mass_fraction_measurements.csv')
elong_data = pd.read_csv('../../data/dai2016_elongation_rate.csv')
data = data[data['growth_rate_hr']>= 0.5]
elong_data = elong_data[elong_data['growth_rate_hr']>= 0.5]

# Load the MCMC inference
opt_samps = pd.read_csv('../../data/optimal_allocation_MCMC_samples.csv')

# %%
gamma_max_samps = opt_samps[opt_samps['parameter']=='gamma_max']
Kd_samps = opt_samps[opt_samps['parameter']=='Kd']
gamma_dist = alt.Chart(gamma_max_samps).mark_bar(color=colors['primary_blue'],  
                                                 opacity=0.75).encode(
                    x=alt.X('value:Q', title='translational efficiency [per hr]',
                            bin=alt.Bin(maxbins=200)),
                    y='count()'
        ).properties(height=100)

Kd_dist = alt.Chart(Kd_samps).mark_bar(color=colors['primary_blue'],  
                                                 opacity=0.75).encode(
                    x=alt.X('value:Q', title='effective dissociation constant [frac. abund.]',
                            bin=alt.Bin(maxbins=200)), 
                    y='count()'
        ).properties(height=100)

gamma_dist & Kd_dist

# %%
percs = [(97.5, 2.5), (87.5, 12.5), (75, 25), (62.5, 37.5), (52.5, 47.5)]
perc_labels = [95, 75, 50, 25, 5]

nu_range = np.linspace(0.01, 10, 300)
fit_df = pd.DataFrame([])
gamma_max = gamma_max_samps['value'].values
Kd = Kd_samps['value'].values
phiO = 0.35
for i, nu in enumerate(nu_range):
    opt_allo_phiR =  growth.model.phi_R_optimal_allocation(gamma_max, nu, Kd, phi_O=0.35)
    opt_allo_lam = growth.model.steady_state_growth_rate(gamma_max, nu, opt_allo_phiR, 1 - phiO - opt_allo_phiR, Kd)
    opt_allo_cAA = growth.model.steady_state_tRNA_balance(nu, 1 - phiO - opt_allo_phiR, opt_allo_lam)
    opt_allo_gamma = growth.model.translation_rate(gamma_max, opt_allo_cAA, Kd)
    avg_phiR = np.median(opt_allo_phiR)
    avg_lam = np.median(opt_allo_lam)
    avg_gamma = np.median(opt_allo_gamma) * 7459/3600
    fit_df = fit_df.append({'nu':nu, 'avg_phiR':avg_phiR, 'avg_lam':avg_lam,
                            'gamma':avg_gamma}, ignore_index=True)
    # for perc, lab in zip(percs, perc_labels):


frac_points = alt.Chart(data, width=300, height=300).mark_point().encode(  
                x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]',
                        scale=alt.Scale(zero=False)),
                y=alt.Y('mass_fraction:Q', title='ribosomal mass fraction'),
                shape=alt.Shape('source:N', title='reference')
)
elong_points = alt.Chart(elong_data, width=300, height=300).mark_point().encode(
                x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]'),
                y=alt.Y('elongation_rate_aa_s:Q', title='elongation rate [AA / s]',
                    scale=alt.Scale(zero=False))
)
frac_avg = alt.Chart(fit_df).mark_line().encode(
                x=alt.X('avg_lam:Q', title='growth rate [per hr]'),
                y=alt.Y('avg_phiR:Q', title='ribosomal mass fraction')

)
elong_avg = alt.Chart(fit_df).mark_line().encode(
              x=alt.X('avg_lam:Q', title='growth_rate_hr:Q'),
              y=alt.Y('gamma:Q', title='elongation rate [AA / s]')
)
(frac_points + frac_avg) | (elong_points + elong_avg)
# %%
