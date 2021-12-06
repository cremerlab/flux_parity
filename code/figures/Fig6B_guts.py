#%%
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz 
import seaborn as sns
colors, palette = growth.viz.matplotlib_style()

# Load the datasets
flow = pd.read_csv('../../data/PhillipsGiller1973_flow.csv')
flow = flow[flow['line'].isin(['Solid'])] #, 'Dotted'])]
gluc = pd.read_csv('../../data/Stephen1983_Fig4.csv')
# %%

fig, ax = plt.subplots(1, 2, figsize=(3.75, 1.75))
# Format axes
ax[0].set_xlabel('time of day [hr]')
ax[0].set_ylabel('flow rate [mL / min]')
ax[1].set_xlabel('time after meal [hr]')
ax[1].set_ylabel('nutrient concentration\nin cecum [mg / mL]')

# Add flow data
ax[0].plot(flow['time_hr'], flow['flow_ml_min'], '-o', color=colors['primary_blue'],
                lw=1, ms=3, markeredgewidth=0.5, zorder=99)

# Add meal markers by hand
ax[0].vlines(8, 0, 10, color=colors['light_black'], linewidth=2, zorder=1000, alpha=0.5)
ax[0].vlines(11, 0, 10, color=colors['light_black'], linewidth=2, zorder=1000, alpha=0.5)
ax[0].vlines(17, 0, 10, color=colors['light_black'], linewidth=2, zorder=1000, alpha=0.5)
ax[0].vlines(20, 0, 10, color=colors['light_black'], linewidth=2, zorder=1000, alpha=0.5)

# Subsample every third point due to over digitizing
_colors = [colors['primary_blue'], colors['primary_purple']]
counter = 0
for g, d in gluc.groupby(['nutrient']):
    ax[1].plot(d['time_from_meal'], d['concentration'], '-o', lw=1, 
                color=_colors[counter], ms=3, markeredgewidth=0.5)
    counter += 1
ax[0].set_ylim([0, 10])
plt.tight_layout()
plt.savefig('../../figures/Fig6B_gut_flow_plots.pdf', bbox_inches='tight')
# %%
