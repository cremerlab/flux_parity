#%%
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import growth.viz
colors, palette = growth.viz.matplotlib_style()

# Load
data = pd.read_csv('../../data/Balarkrishnan_2021_wt_shift_RNAseq_classified.csv')

# %%
steadystate = data[data['phase'] != 'shift']
steadystate_grouped = steadystate.groupby(['phase', 'cog_letter']).sum().reset_index()
# %%
steadystate_grouped
steadystate_grouped['idx'] = 0
steadystate_grouped.loc[steadystate_grouped['phase']=='ss_postshift', 'idx'] = 1

# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))

for g, d in steadystate_grouped.groupby(['cog_letter']):
    ax.plot(d['idx'], d['fraction'], '-o', label=g)
ax.legend()
# %%
sector = steadystate.groupby(['phase', 'sector']).sum().reset_index()
sector['idx'] = 0
sector.loc[sector['phase']=='ss_postshift', 'idx'] = 1



# %%
fig, ax = plt.subplots(1, 1, figsize=(6, 4))
for g, d in sector.groupby(['sector']):
    ax.plot(d['idx'], d['fraction'], '-o', label=g)
ax.legend()
# %%

steadystate[steadystate['gene'].isin(['aceA', 'aceB'])]

# %%
sector

# %%
phiRb_preshift = 0.292836
phiRb_postshift = 0.158632
lam_preshift = 0.94
lam_postshift = 0.38

def compute_numax(lam, phiRb, phiO, gamma_max=9.65, Kd=0.01):
    numer = lam * (Kd * lam + gamma_max * phiRb - lam)
    denom = (gamma_max * phiRb - lam) * (1 - phiRb - phiO)
    return numer/denom

nu_preshift = compute_numax(lam_preshift, phiRb_preshift, 0.3)
nu_postshift = compute_numax(lam_postshift, phiRb_postshift, 0.3)

# %%
