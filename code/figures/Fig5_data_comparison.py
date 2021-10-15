#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import growth.model 
import growth.viz 
import seaborn as sns 
color, _ = growth.viz.matplotlib_style()


# Load the data sets
mass_frac = pd.read_csv('../../data/ribosomal_mass_fractions.csv')
elong_rate = pd.read_csv('../../data/peptide_elongation_rates.csv')

#%%
# Set up the figure canvas
fig, ax = plt.subplots(2, 3, figsize=(5.2, 4))
ax[0,0].axis('off')
ax[1,0].axis('off')

# Add labels
for i in range(2):
    ax[i,1].set(ylabel='ribosomal mass fraction $\phi_{Rb}$',
            xlabel='growth rate $\mu$ [hr$^{-1}$]')
    ax[i,2].set(ylabel='translational efficiency $\gamma$ [hr$^{-1}$]',
             xlabel='growth rate $\mu$ [hr$^{-1}$]')


# Set ranges
ax[0, 1].set(ylim=[0, 0.3], xlim=[0, 2])
ax[1, 1].set(ylim=[0, 0.35], xlim=[0, 0.7])
# Plot mass fraction
for g, d in mass_frac.groupby('organism'):
    print(g)
    if g == 'Escherichia coli':
        _ax = ax[0,1]
    else:
        _ax = ax[1, 1]
    for _g, _d in d.groupby(['source']): 
        _ax.plot(_d['growth_rate_hr'], _d['mass_fraction'], 'o', ms=3,
                label=_g)
ax[0, 1].legend()
ax[1, 1].legend()
plt.tight_layout()
# %%
