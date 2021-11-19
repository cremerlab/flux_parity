#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns 
import growth.viz
import growth.model
import scipy.integrate
colors, palette=  growth.viz.matplotlib_style()

# Load the constants
const = growth.model.load_constants()
# nu_max = 1.85 
nu_max = 4.5

# Define the allocation parameters
phi_O = 0.55
opt_phiRb = growth.model.phiRb_optimal_allocation(const['gamma_max'],
                                                  nu_max,
                                                  const['Kd_cpc'],
                                                  phi_O)
# Change to iterate over phiRB
phi_Rb = np.array([opt_phiRb]) 
phi_Mb = 1 - phi_Rb - phi_O

# Define starting masses
M0 = 0.001 * const['OD_conv']
MR_0 = 0.2 * M0
MP_0 = (1 - phi_O - opt_phiRb) * M0

# Define starting concentrations
cpc_0 = 0.01 # in abundance units
cnt_0 = 0.01

# Set up the time range of the integration
time_range = np.linspace(0, 12, 300)
dfs = []
for i, phi in enumerate(phi_Rb):
    M_Rb = phi * M0
    M_Mb = phi_Mb[i] * M0

    # Pack parameters and arguments
    params = [M0, M_Rb, M_Mb, cpc_0, cnt_0] 
    args = (const['gamma_max'], 
        nu_max, 
        const['Y'],
        phi,
        phi_Mb[i],
        const['Kd_cpc'],
        const['Kd_cnt'])


    # Perform the integration
    out = scipy.integrate.odeint(growth.model.self_replicator, 
                                params, time_range, args=args)

    # Pack the integration output into a tidy dataframe
    df = pd.DataFrame(out, columns=['M', 'M_Rb', 'M_Mb', 'c_pc', 'c_nt'])
    df['rel_biomass'] = df['M'].values / M0
    df['time'] = time_range
    df['gamma'] = const['gamma_max'] * (df['c_pc'].values / (df['c_pc'].values + const['Kd_cpc']))
    df['time'] = time_range
    df['phi_Rb'] = phi
    dfs.append(df)
df = pd.concat(dfs, sort=False)

# Instantiate figure and label/format axes
fig, ax = plt.subplots(1, 3, figsize=(6, 2), sharex=True)
for a in ax:
    a.set_xlabel('time [hr]')

# Add labels
ax[0].set_ylabel('approximate\noptical density [a.u.]')
ax[0].set_ylim([1E-3, 5])
ax[1].set_ylim([0, 22])
# ax[2].set_ylim([0, 3.2])
ax[1].set_ylabel(r'$c_{pc}\, /\,  K_D^{c_{pc}}$' + '\nprecursor concentration')
ax[2].set_ylabel(r'$c_{nt}\, /\, K_D^{c_{nt}}$' + '\nprecursor concentration')
ax[0].set_yscale('log')

# Add titles
ax[0].set_title('biomass dynamics')
ax[2].set_title('precursor dynamics')
ax[1].set_title('nutrient dynamics')
cmap = [colors['primary_black']]

# Add curve
count = 0
for g, d in df.groupby('phi_Rb'):
    ax[0].plot(d['time'], d['M'].values / const['OD_conv'], '-', lw=1, color=cmap[count])
    ax[2].plot(d['time'], d['c_pc'] / const['Kd_cpc'], '-', lw=1, color=cmap[count])
    ax[1].plot(d['time'], d['c_nt']  / const['Kd_cnt'], '-', lw=1, color=cmap[count])
    count += 1

# Find the steady state region
dd_cpc = np.diff(np.diff(d['c_pc'].values))
where = np.where(np.round(dd_cpc, decimals=4) == 0)
ss_begin = time_range[where[0][0] - 2]
ss_end = time_range[np.where(np.diff(where[0]) > 1)[0]- 3]

# Add a shaded region in all three plots for the exponential regime
for a in ax:
    a.fill_betweenx([a.get_ylim()[0], a.get_ylim()[1]], ss_begin, ss_end, 'k', alpha=0.25)
plt.tight_layout()
plt.savefig('../../figures/Fig2_dynamics_plots.pdf', bbox_inches='tight')
# %%
