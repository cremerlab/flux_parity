#%%
import pandas as pd 
import growth.viz 
import matplotlib.pyplot as plt
import scipy.stats
colors, palette = growth.viz.matplotlib_style()
mapper = growth.viz.load_markercolors()

# Load the data 
data = pd.read_csv('../data/supplement_figure_data/Schmidt2016_proteomics.csv')

#%%
# Compute mass fractions of each gene. 
dfs = []
for g, d in data.groupby(['dataset_name', 'growth_rate_hr', 'condition']):
    tot = d['fg_per_cell'].sum()
    d['total_mass'] = tot
    d['mass_fraction'] = d['fg_per_cell'] / tot
    dfs.append(d)
data = pd.concat(dfs, sort=False)

# Compute the pearson corrrelation for each gene
dfs = []
for g, d in data.groupby(['dataset_name', 'gene_name']):
    if len(d) >= 2:
        corr = scipy.stats.pearsonr(d['growth_rate_hr'], d['mass_fraction'])[0]
        d['pearson'] = corr
        if corr <= -0.5:
            d['trend'] = 'negative correlation' 
        elif corr >= 0.5:
            d['trend'] = 'positive correlation'
        else:
            d['trend'] = 'no or weak correlation'
        dfs.append(d) 
data = pd.concat(dfs, sort=False)


# Assign COG classification by my breakdown
ribos = ['J']
metabs = ['P', 'H', 'F', 'E', 'G', 'C']
data['cat'] = 'others'
data.loc[data['cog_letter'].isin(ribos), 'cat'] = 'ribosomal'
data.loc[data['cog_letter'].isin(metabs), 'cat'] = 'metabolic'

# Generate aggregations
trend_grouped = data.groupby(['growth_rate_hr', 'condition', 'trend', 'dataset_name']).sum().reset_index()
cog_grouped = data.groupby(['growth_rate_hr', 'condition', 'cat', 'dataset_name']).sum().reset_index()
#%%
# Select one condition and show the composition of the 'no correlation' factor 
# by internal COG association
composition = data[(data['growth_rate_hr']==0.58) & 
                   (data['trend'] == 'no or weak correlation')]
composition_grouped = composition.groupby(['cat']).sum().reset_index()
composition_grouped['frac'] = composition_grouped['fg_per_cell'] / composition['fg_per_cell'].sum()
#%%
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
cog_color_mapper = {'ribosomal': colors['primary_gold'],
                    'metabolic': colors['primary_purple'],
                    'others': colors['primary_black']}
trend_color_mapper = {'positive correlation': colors['primary_blue'],
                      'negative correlation': colors['primary_red'],
                      'no or weak correlation': colors['primary_black']}
for g, d in trend_grouped.groupby(['trend', 'dataset_name']):
    idx = 'Schmidt et al., 2016'
    ax[1].plot(d['growth_rate_hr'], d['mass_fraction'], marker=mapper[idx]['m'],
                linestyle='none', markeredgecolor='k',  
                markerfacecolor=trend_color_mapper[g[0]],
                alpha=0.75, ms=4)

for g, d in cog_grouped.groupby(['cat', 'dataset_name']):
    if g[1] == 'Li et al. 2014':
        idx = 'Li et al., 2014'
    else:
        idx = 'Schmidt et al., 2016'
    ax[0].plot(d['growth_rate_hr'], d['mass_fraction'], marker=mapper[idx]['m'],
                linestyle='none', markeredgecolor='k',  
                markerfacecolor=cog_color_mapper[g[0]],
                alpha=0.75, ms=4)

for a in ax[:2]:
    a.set(xlabel='growth rate\n$\lambda$ [hr$^{-1}$]',
          ylim=[0, 0.8],
          xlim = [0, 2],
          ylabel='mass fraction of proteome')

plt.tight_layout()
plt.savefig('../figures/supplement_text/plots/FigS3_phiO_composition_plots.pdf', bbox_inches='tight')
# %%
# Plot the pies.
# Select one condition and show the composition of the 'no correlation' factor 
# by internal COG association
no_trend = data[(data['growth_rate_hr']==0.58) & 
                   (data['trend'] == 'no or weak correlation')]
positive = data[(data['growth_rate_hr']==0.58) & 
                   (data['trend'] == 'positive correlation')]
negative = data[(data['growth_rate_hr']==0.58) & 
                   (data['trend'] == 'negative correlation')]
others_grouped = no_trend.groupby(['cat']).sum().reset_index()
others_grouped['frac'] = others_grouped['fg_per_cell'] / no_trend['fg_per_cell'].sum()
negative_grouped = negative.groupby(['cat']).sum().reset_index()
negative_grouped['frac'] = negative_grouped['fg_per_cell'] / negative['fg_per_cell'].sum()
positive_grouped = positive.groupby(['cat']).sum().reset_index()
positive_grouped['frac'] = positive_grouped['fg_per_cell'] / positive['fg_per_cell'].sum()
fig, ax = plt.subplots(1, 3, figsize=(6, 2))
ax[0].pie(others_grouped['frac'].values, colors=[colors['primary_purple'], colors['primary_black'], colors['primary_gold']])
ax[1].pie(negative_grouped['frac'].values, colors=[colors['primary_purple'], colors['primary_black'], colors['primary_gold']])
ax[2].pie(positive_grouped['frac'].values, colors=[colors['primary_purple'], colors['primary_black'], colors['primary_gold']])
plt.savefig('../figures/supplemental_text/plots/FigS3_phiO_composition_pies.pdf', bbox_inches='tight')
# %%
