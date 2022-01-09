#%%
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import tqdm
import growth.viz 
import growth.model
import scipy.integrate
import growth.integrate
colors, palette = growth.viz.matplotlib_style()
const = growth.model.load_constants()

nu_range = np.linspace(0.2, 20, 200)

df = pd.DataFrame([])
for i, nu in enumerate(tqdm.tqdm(nu_range)):
    # Pack args
    args = {'gamma_max': const['gamma_max'],
            'Kd_TAA': const['Kd_TAA'],
            'Kd_TAA_star': const['Kd_TAA_star'],
            'kappa_max': const['kappa_max'],
            'tau': const['tau'],
            'phi_O': const['phi_O'],
            'nu_max': nu}
    out = growth.integrate.equilibrate_FPM(args)

    TAA = out[-2]
    TAA_star = out[-1]
    ratio = TAA_star / TAA
    phiRb = (1 - const['phi_O']) * (ratio / (ratio + const['tau']))
    gamma = const['gamma_max'] * (TAA_star / (TAA_star + const['Kd_TAA_star']))
    df = df.append({'nu_max': nu,
                    'ratio': ratio,
                    'phiRb': phiRb,
                    'gamma': gamma,
                    'growth_rate': gamma * phiRb},
                    ignore_index=True)
# %%
xs = np.array([0.1343570057581570,
0.18426103646833000,
0.20345489443378100,
0.21113243761996200,
0.23416506717850300,
0.26487523992322500,
0.33013435700575800,
0.345489443378119,
0.3416506717850290,
0.4069097888675620,
0.42994241842610400,
0.43378119001919400,
0.4990403071017280,
0.5834932821497120,
0.5758157389635320,
0.7523992322456810,
1.001919385796550,
1.1938579654510600,
1.45873320537428,
1.8426103646833000])

ys = np.array([0.4516129032258080,
0.9216589861751160,
1.0599078341013800,
1.3364055299539200,
1.1428571428571400,
1.225806451612900,
1.3640552995391700,
1.612903225806450,
1.668202764976960,
1.9447004608294900,
1.6958525345622100,
1.9170506912442400,
2.193548387096770,
2.2764976958525300,
2.8847926267281100,
3.35483870967742,
4.184331797235020,
4.792626728110600,
5.870967741935480,
7.419354838709680])


fig, ax = plt.subplots(1, 1)
ind = np.where(np.round(df['growth_rate'].values, decimals=2) == 0.96)[0][0]
df['rel'] = (1/df['ratio'].values) / (1/df['ratio'].values[ind])
ax.plot((const['gamma_max'] / df['gamma'].values) - 1, df['rel'])
ax.plot(xs, ys, 'o')


# %%
xs = np.array([0.11910112359550600,
0.3240449438202250,
0.3510112359550560,
0.33303370786516900,
0.36359550561797800,
0.42651685393258400,
0.6476404494382020,
0.6764044943820220,
0.955056179775281])

ys = np.array([2.883563402889250,
2.1656179775280900,
1.8937078651685400,
1.6797431781701400,
1.664911717495990,
1.8495024077046500,
1.288410914927770,
1.102182985553770,
1.0114927768860400])
plt.plot(df['growth_rate'], df['rel'], 'k-', lw=2)
plt.plot(xs, ys, 'o')
plt.xlabel('growth rate')
plt.ylabel('relative ppGpp')
plt.ylim([0, 4])
# %%
