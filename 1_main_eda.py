import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

from global_vars import path_data
sns.set_style({'axes.grid' : False})
sns.set_context('paper')
pd.set_option("display.precision", 2)


#### load original data
data = pd.read_csv(os.path.join(path_data,r'Tarifierung_RI_2017.csv'),  delimiter=';'  )
N = len(data)

# summary of categorical features
f1_level, f1_counts = np.unique(data['ZahlweiseInkasso'], return_counts=True)
f2_level, f2_counts = np.unique(data['GeschlechtVP1'], return_counts=True)
f3_level, f3_counts = np.unique(data['RauchertypVP1'], return_counts=True)

df = pd.DataFrame(data = None)
df['ZahlweiseInkasso'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f1_level, f1_counts)]
df['GeschlechtVP1'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f2_level, f2_counts)]+['', '']
df['RauchertypVP1'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f3_level, f3_counts)]+['', '']



print(data.describe(include=None).to_latex())
print(df.to_latex())

data['Beginnjahr'] = data['Beginnjahr'] .astype('int')
data.columns = ['year', 'month', 'payment', 'gender', 'smoker', 'age', 'n', 't', 'sum insured', 'premium']

# set some plotting parameters globally
# parameters = {'axes.labelsize': 16, 'xtick.labelsize':14, 'ytick.labelsize': 14, 'legend.fontsize': 14, 'axes.titlesize': 16, 'figure.titlesize': 18}
# plt.rcParams.update(parameters)

_, ax = plt.subplots(3,3)
ax = ax.flatten()
ax[-1].remove()
ax[-2].remove()
for k, e in enumerate(['year', 'month', 'age', 'n', 't', 'sum insured', 'premium']):
    ax[k].hist(data[e], bins = 100, weights = np.zeros(N) + 1. / N, color = 'gray')#, font = 'large')
    ax[k].set_xlabel(e)
    if e == 'year':
        ax[k].set_xticks([2015, 2016])
    if e == 'sum insured':
        ax[k].set_xlabel('S')
        ax[k].set_xticks([0,5e5,1e6])

# data.hist(bins=100, grid = False, weights=np.zeros(N) + 1. / N, color = 'gray')
plt.tight_layout()
plt.savefig(os.path.join(path_data,r'data_marginal_dists.eps'))
plt.savefig(os.path.join(path_data,r'data_marginal_dists.png'), dpi=400)
plt.close()

# print(data.corr('pearson'))
# print(data.corr('pearson').to_latex())

# sns.heatmap(data.corr('pearson'))
# plt.show()