#%%
import numpy as np 
import pandas as pd 
import growth.viz 
import growth.model
import altair as alt
from altair_saver import save
colors, palette = growth.viz.altair_style()

frac_data = pd.read_csv('../../data/mass_fraction_compiled.csv')
frac_data = frac_data[frac_data['growth_rate_hr'] >= 0.5]
elong_data = pd.read_csv('../../data/dai2016_elongation_rate.csv')
elong_data = elong_data[elong_data['growth_rate_hr'] >= 0.5]
#%%
gamma_max = 17.1 * 3600 / 7459
nu_max = np.linspace(0.1, 10, 300)
fa = 1
Kd =  2E-3
phi_O = 0.35
opt_phi_R = growth.model.optimal_phi_R(gamma_max, nu_max, Kd, phi_O, f_a=fa)
opt_phi_P = 1 - phi_O - opt_phi_R
growth_rate = growth.model.growth_rate(nu_max, gamma_max, opt_phi_R, opt_phi_P, Kd, f_a=fa)
cAA = growth.model.tRNA_balance(nu_max, opt_phi_P, growth_rate)
gamma = growth.model.translation_rate(gamma_max, cAA, Kd)
optimal_df = pd.DataFrame([])
optimal_df['nu_max'] = nu_max
optimal_df['phi_R'] = opt_phi_R
optimal_df['growth_rate'] = growth_rate
optimal_df['cAA'] = cAA
optimal_df['gamma'] = gamma * 7459/3600



# Data plots
data_plot = alt.Chart(frac_data, width=300, height=300).mark_point(size=80, opacity=0.45).encode(
            x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]',
                    scale=alt.Scale(domain=(0.5, 2.15))),
            y=alt.Y('mass_fraction:Q', title='allocation to translation',
                    axis=alt.Axis(format='%')),
            color =alt.Color('source:N', title='data source')
)

elong_plot = alt.Chart(elong_data, width=300, height=300
                       ).mark_point(size=80, opacity=0.45).encode(
                        x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]',
                                scale=alt.Scale(domain=(0.5, 2.1))),
                        y=alt.Y('elongation_rate_aa_s:Q', title='elongation rate [AA/s]')
                       )


opt_base = alt.Chart(optimal_df, width=300, height=300).encode(
            x=alt.X('growth_rate:Q', title='growth rate [per hr]'))

opt_phiR = opt_base.mark_line(color=colors['black'],size=2, clip=True).encode(
            y=alt.Y('phi_R:Q', title='allocation to translation',
                    axis=alt.Axis(format='%'))
)
opt_gamma = opt_base.mark_line(color=colors['black'], size=2, clip=True).encode(
            x=alt.X('growth_rate:Q', title='growth rate [per hr]'),
            y=alt.Y('gamma:Q', title='elongation rate [AA/s]')
)
layer = (data_plot + opt_phiR)  | (elong_plot + opt_gamma)
layer
# save(layer, './optimal_allocation.pdf')
# %%


# %%

# %%
