#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import growth.viz 
colors, _ = growth.viz.matplotlib_style()
palette = sns.color_palette('mako', n_colors=8)

# Load the elongation data
elong_data = pd.read_csv('../../data/peptide_elongation_rates.csv')
ecoli_elong = elong_data[elong_data['organism']=='Escherichia coli']
yeast_elong = elong_data[elong_data['organism']!='Escherichia coli']

# Load the mass fraction data 
yeast_mass = pd.read_csv('../../data/yeast_mass_fraction.csv')
ecoli_mass = pd.read_csv('../../data/collated_mass_fraction_measurements.csv')
# %%
fig, ax = plt.subplots(2, 2, figsize=(4, 4))
for a in ax.ravel():
    a.xaxis.set_tick_params(labelsize=6)
    a.yaxis.set_tick_params(labelsize=6)
    a.set_xlabel('growth rate [hr$^{-1}$]')

count = 0
markers = ['o', 's', 'X', 'v', '^', 'd']
for g, d in ecoli_mass.groupby('source'):
   ax[0,0].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', color=palette[count],
                ms=4, alpha=0.75, label=g, linestyle='none') 
   count += 1 

count = 0
markers = ['o', 's', 'X', 'v', '^', 'd']
for g, d in yeast_mass.groupby('source'):
   ax[1,0].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', color=palette[count],
                ms=4, alpha=0.75, label=g, linestyle='none') 
   count += 1 


count = 0
for g, d in  ecoli_elong.groupby('source'):
    ax[0, 1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o',
                color=palette[count], ms=4, alpha=0.75)
    count +=1 

for g, d in  yeast_elong.groupby('source'):
    ax[1, 1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o',
                color=palette[count], ms=4, alpha=0.75)
    count +=1 


for i in range(2):
    ax[i, 0].set_ylim([0.01, 0.4])
# ax[0,0].legend()
#%%
# Set up the various plots
ecoli_elong_plot = alt.Chart(ecoli_elong).mark_point().encode(
                   x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]'),
                   y=alt.Y('elongation_rate_aa_s:Q', title='elongation rate [AA/s]'),
                   color=alt.Color('source:N', title='data source'),
                   shape=alt.Shape('source:N', title='data_source')
)

ecoli_mass_fraction = alt.Chart(ecoli_mass).mark_point().encode(
                x=alt.X('growth_rate_hr:Q)
)

points = alt.Chart(data, width=300, height=300).mark_point().encode(
            x=alt.X('growth_rate_hr:Q', title='growth rate [per hr]'),
            y=alt.Y('elongation_rate_aa_s:Q', title='peptide chain elongation rate [AA /s]'),
            color=alt.Color('source:N', title='Primary Source')
).facet(row='organism:N').resolve_scale( x='independent', 
                                        y='independent')
save(points, '../../figures/elongation_rates.pdf')
# %%
