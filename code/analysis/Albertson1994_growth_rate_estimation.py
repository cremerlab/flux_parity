#%%
import numpy as np 
import pandas as pd 
import scipy.stats
import growth.viz
import matplotlib.pyplot as plt 
colors, palette = growth.viz.matplotlib_style()
# %%

# Load the data and restrict
data = pd.read_csv('../../data/Albertson1994_Fig1A.csv')
data = data[data['region']=='exponential']

# COmpute the log and do the fit
data['log_od'] = np.log(data['od_420nm'].values)

# Perform the fit
popt = scipy.stats.linregress(data['time_hr'].values, data['log_od'].values)

# Make a plot to make sure it looks okay.
time = np.linspace(0, 1.75, 100)
fit = np.exp(popt[1] + popt[0] * time)

fig, ax = plt.subplots(1, 1)
ax.set_yscale('log')
ax.plot(data['time_hr'], data['od_420nm'], '.')
ax.plot(time, fit, '-')

# %%
