#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import growth.viz 
import growth.model
colors, palette = growth.viz.matplotlib_style()
# palette = sns.color_palette('crest', n_colors=8)

# Load the elongation data
elong_data = pd.read_csv('../../data/peptide_elongation_rates.csv')
ecoli_elong = elong_data[elong_data['organism']=='Escherichia coli']
yeast_elong = elong_data[elong_data['organism']!='Escherichia coli']

# Load the mass fraction data 
yeast_mass = pd.read_csv('../../data/yeast_mass_fraction.csv')
ecoli_mass = pd.read_csv('../../data/collated_mass_fraction_measurements.csv')


# %% For each system, define the parameters
nu_max = np.linspace(0, 5, 200)
ecoli_params = {'nu_max':nu_max,
                'gamma_max': 20 * 3600 / 7459,
                'Kd':0.015}
yeast_params = {'nu_max':nu_max,
                'gamma_max': 10 * 3600 / 11984,
                'Kd':0.12}
 
ecoli_phiR = growth.model.phi_R_optimal_allocation(gamma_max=ecoli_params['gamma_max'],
                                                    nu_max=ecoli_params['nu_max'],
                                                    Kd=ecoli_params['Kd'],
                                                    phi_O = 0)
ecoli_lam = growth.model.steady_state_growth_rate(ecoli_params['gamma_max'],
                                                ecoli_params['nu_max'],
                                                phi_R = ecoli_phiR,
                                                phi_P = 1 - ecoli_phiR,
                                                Kd=ecoli_params['Kd'],
                                                )
ecoli_cAA = growth.model.sstRNA_balance(nu_max=ecoli_params['nu_max'],
                                        phi_P = (1-ecoli_phiR),
                                        gamma_max=ecoli_params['gamma_max'],
                                        phi_R=ecoli_phiR,
                                        Kd=ecoli_params['Kd'])

ecoli_gamma = growth.model.translation_rate(gamma_max=ecoli_params['gamma_max'], 
                                            c_AA = ecoli_cAA,
                                            Kd=ecoli_params['Kd'])
ecoli_gamma *= 7459 / 3600

yeast_phiR = growth.model.phi_R_optimal_allocation(gamma_max=yeast_params['gamma_max'],
                                                    nu_max=yeast_params['nu_max'],
                                                    Kd=yeast_params['Kd'],
                                                    phi_O = 0)
yeast_lam = growth.model.steady_state_growth_rate(yeast_params['gamma_max'],
                                                yeast_params['nu_max'],
                                                phi_R = yeast_phiR,
                                                phi_P = 1 - yeast_phiR,
                                                Kd=yeast_params['Kd'],
                                                )
yeast_cAA = growth.model.sstRNA_balance(nu_max=yeast_params['nu_max'],
                                        phi_P = (1-yeast_phiR),
                                        gamma_max=yeast_params['gamma_max'],
                                        phi_R=yeast_phiR,
                                        Kd=yeast_params['Kd'])

yeast_gamma = growth.model.translation_rate(gamma_max=yeast_params['gamma_max'], 
                                            c_AA = yeast_cAA,
                                            Kd=yeast_params['Kd'])
yeast_gamma *= 11984 / 3600






fig, ax = plt.subplots(2, 2, figsize=(4, 4))

# Format the axes
for a in ax.ravel():
    a.xaxis.set_tick_params(labelsize=6)
    a.yaxis.set_tick_params(labelsize=6)
    a.set_xlabel('growth rate [hr$^{-1}$]')


# Plot the E coli mass frac data
count = 0
markers = ['o', 's', 'X', 'v', '^', 'd']
for g, d in ecoli_mass.groupby('source'):
   ax[0,0].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', color=palette[count],
                ms=4, alpha=0.75, label=g, linestyle='none') 
   count += 1 

# Plot the E coli mass frac theory
ax[0, 0].plot(ecoli_lam, ecoli_phiR, 'k-', lw=1)



# Plot the yeast mass frac data
count = 0
markers = ['o', 's', 'X', 'v', '^', 'd']
for g, d in yeast_mass.groupby('source'):
   ax[1,0].plot(d['growth_rate_hr'], d['mass_fraction'], 'o', color=palette[count],
                ms=4, alpha=0.75, label=g, linestyle='none') 
   count += 1 

# Plot the yeast coli mass frac theory
ax[1, 0].plot(yeast_lam, yeast_phiR, 'k-', lw=1)



# Plot the E coli translation rate data
count = 0
for g, d in  ecoli_elong.groupby('source'):
    ax[0, 1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o',
                color=palette[count], ms=4, alpha=0.75)
    count +=1 

# Plot the E coli translation rate theory
ax[0, 1].plot(ecoli_lam, ecoli_gamma, 'k-', lw=1)

# Plot the yeast translation rate data
for g, d in  yeast_elong.groupby('source'):
    ax[1, 1].plot(d['growth_rate_hr'], d['elongation_rate_aa_s'], 'o',
                color=palette[count], ms=4, alpha=0.75)
    count +=1 

# Plot the yeast translation rate theory
ax[1, 1].plot(yeast_lam, yeast_gamma, 'k-', lw=1)

    
# ax[0,0].legend()
# Adjust the axis limits
ax[0, 0].set_xlim([0, 2])
ax[0, 1].set_xlim([0, 2.5])
ax[0, 0].set_ylim([0, 0.3])
ax[1, 0].set_xlim([0, 0.9])
ax[1, 1].set_xlim([0, 0.9])
ax[1, 0].set_ylim([0, 0.4])
ax[1, 1].set_ylim([0, 12])

# Add titles and things

for i in range(2):
    ax[i, 0].set_ylabel('ribosomal mass fraction', fontsize=8)
    ax[i, 1].set_ylabel('elongation rate [AA/s]', fontsize=8)
    ax[0, i].set_title('E. coli', fontsize=8, fontstyle='italic')
    ax[1, i].set_title('S. cerevisiae', fontsize=8, fontstyle='italic')
plt.tight_layout()
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
