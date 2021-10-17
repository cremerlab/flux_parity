#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.viz
colors, palette = growth.viz.matplotlib_style()


# Load the proteomic data 
data = pd.read_csv('../../data/compiled_absolute_measurements.csv')
data.head()
# %%
# Define the categorization
catalog = {
    'protein_synthesis':  ['J'],
    'metabolism': ['P', 'H', 'F', 'E', 'G', 'C'],
    'other': ['X', 'O', 'U', 'W', 'Z', 'N', 'M', 'T', 'V',
              'Y', 'D', 'B', 'L', 'K', 'A', 'R', 'S',
              'Not Found', 'Not Assigned']
}

for k, v in catalog.items():
    data.loc[data['cog_letter'].isin(v), 'sector'] = k
# %%
# Group and compute the mass fraction
sector_df = pd.DataFrame([])
for g, d in data.groupby(['dataset_name', 'condition', 'growth_rate_hr']):
    tot_mass = d['fg_per_cell'].sum()
    for _g, _d in d.groupby(['sector']):
        frac = _d['fg_per_cell'].sum() / tot_mass

        sector_df = sector_df.append({'sector':_g,
                                      'total_proteome_mass': tot_mass,
                                      'sector_mass': _d['fg_per_cell'].sum(),
                                      'mass_fraction': frac,
                                      'dataset': g[0],
                                      'condition':g[1],
                                      'growth_rate_hr':g[2]},
                                      ignore_index=True)

# %%
fig, ax = plt.subplots(1,1)
counter = 0
sector_colors = {'protein_synthesis':colors['primary_gold'],
                 'metabolism': colors['primary_purple'],
                 'other': colors['light_black']}
markers = {'Li et al. 2014': 's',
           'Schmidt et al. 2016': 'o',
           'Valgepea et al. 2013': 'X',
           'Peebo et al. 2015': 'D'}
for g, d in sector_df.groupby(['dataset', 'sector']):
    ax.plot(d['growth_rate_hr'], d['mass_fraction'], marker=markers[g[0]], color=sector_colors[g[1]], linestyle='none')
ax.set_xlabel('growth rate [hr$^{-1}$]')
ax.set_ylabel('proteome fraction')
plt.tight_layout()
plt.savefig('../../figures/FigSX_sector_plots.pdf', bbox_inches='tight')
# %%
