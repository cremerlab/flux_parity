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
nu_max = 0.75

# Define the allocation parameters
phi_O = 0.25
opt_phiRb = growth.model.phi_R_optimal_allocation(const['gamma_max'],
                                                  nu_max,
                                                  const['Kd_cpc'],
                                                  phi_O)
phi_Rb = np.array([opt_phiRb]) #np.linspace(0.2, 1, 7) * opt_phiRb
phi_Mb = 1 - phi_Rb - phi_O

# Define starting masses
M0 = 0.1 * const['OD_conv']
MR_0 = phi_Rb * M0
MP_0 = phi_Mb * M0

# Define starting concentrations
cpc_0 = 0.01 # in abundance units
cnt_0 = 0.05

#%%
# Set up the time range of the integration
time_range = np.linspace(0, 25, 300)
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
ax[0].set_ylabel(r'$M(t)\, / \, M(t=0)$')
ax[1].set_ylabel(r'$c_{pc}(t)\, /\,  K_D^{c_{pc}}$')
ax[2].set_ylabel(r'$c_{nt}(t)\, /\, c_{nt}(t=0)$')
ax[0].set_yscale('log')
# ax[1].set_yscale('log')


# Add titles
ax[0].set_title('biomass dynamics')
ax[1].set_title('precursor dynamics')
ax[2].set_title('nutrient dynamics')

# cmap = [colors['dark_gold'], colors['gold'], colors['primary_gold'], colors['light_gold'], colors['pale_gold']]
# cmap = sns.color_palette('YlOrBr_r', n_colors=10)[:-3]
cmap = [colors['primary_black']]

count = 0
for g, d in df.groupby('phi_Rb'):
    ax[0].plot(d['time'], d['rel_biomass'], '-', lw=1, color=cmap[count])
    ax[1].plot(d['time'], d['c_pc'] / const['Kd_cpc'], '-', lw=1, color=cmap[count])
    ax[2].plot(d['time'], d['c_nt']  / cnt_0, '-', lw=1, color=cmap[count])
    count += 1
plt.tight_layout()
plt.savefig('../../figures/Fig2_dynamics_plots.pdf', bbox_inches='tight')
# %%
