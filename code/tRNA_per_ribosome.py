# %%
import numpy as np
import pandas as pd
import growth.viz
import growth.integrate
import growth.model
import tqdm
import matplotlib.pyplot as plt
cor, pal = growth.viz.matplotlib_style()
color_mapper = growth.viz.load_markercolors()
data = pd.read_csv('../data/source_data/tRNA_abundances.csv')
const = growth.model.load_constants()

nu_max_range = np.linspace(0.5, 40, 200)
args = {'gamma_max': const['gamma_max'],
        'Kd_TAA': const['Kd_TAA'],
        'Kd_TAA_star': const['Kd_TAA_star'],
        'kappa_max': const['kappa_max'],
        'tau': const['tau'],
        'phi_O': const['phi_O']
        }
tRNA_per_ribosome = np.zeros(len(nu_max_range))
growth_rate = np.zeros(len(nu_max_range))
for i, nu in enumerate(tqdm.tqdm(nu_max_range)):
    args['nu_max'] = nu
    M, M_Rb, M_Mb, TAA, TAA_star = growth.integrate.equilibrate_FPM(args)
    ratio = TAA_star / TAA
    tot_tRNA = (TAA + TAA_star) * M
    N_Rb = M_Rb / const['m_Rb']
    tRNA_per_ribosome[i] = tot_tRNA / N_Rb
    growth_rate[i] = (M_Rb/M) * const['gamma_max'] * \
        (TAA_star / (TAA_star + const['Kd_TAA_star']))

# %%

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
for g, d in data.groupby(['source']):
    ax.plot(d['growth_rate_hr'], d['tRNA_per_ribosome'], linestyle='-',
            marker=color_mapper[g]['m'], lw=1, color=color_mapper[g]['c'],
            markerfacecolor=color_mapper[g]['c'],
            alpha=0.75, markeredgecolor='k', markeredgewidth=0.5,
            label=g
            )
ax.plot(growth_rate, tRNA_per_ribosome, '--', color=cor['primary_red'], lw=3,
        label='flux-parity prediction')
ax.legend(fontsize=5.5)
ax.set_ylim(5, 20)
ax.set_xlim(0.1, 2.75)
ax.set_xlabel('growth rate [hr$^{-1}$]')
ax.set_ylabel('tRNA molecules per ribosome')
plt.savefig('../figures/FigRX_tRNA_per_ribosome.pdf')
