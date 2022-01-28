'''
Perform Exploratory Data Analysis
'''

from turtle import color
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

# numeric features
# print(data.describe(include=None))

# summary of categorical features
f1_level, f1_counts = np.unique(data['ZahlweiseInkasso'], return_counts=True)
f2_level, f2_counts = np.unique(data['GeschlechtVP1'], return_counts=True)
f3_level, f3_counts = np.unique(data['RauchertypVP1'], return_counts=True)

df = pd.DataFrame(data = None)
df['ZahlweiseInkasso'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f1_level, f1_counts)]
df['GeschlechtVP1'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f2_level, f2_counts)]+['', '']
df['RauchertypVP1'] = [f'{lvl} ({c/N: .2f} )' for lvl, c in zip(f3_level, f3_counts)]+['', '']

# print(df)


print(data.describe(include=None).to_latex())

print(df.to_latex())

data['Beginnjahr'] = data['Beginnjahr'] .astype('int')
data.columns = ['year', 'month', 'payment', 'gender', 'smoker', 'age', 'n', 't', 'sum insured', 'premium']
print(data.columns)

_, ax = plt.subplots(3,3)
ax = ax.flatten()
ax[-1].remove()
ax[-2].remove()
for k, e in enumerate(['year', 'month', 'age', 'n', 't', 'sum insured', 'premium']):
    ax[k].hist(data[e], bins = 100, weights = np.zeros(N) + 1. / N, color = 'gray')#, font = 'large')
    ax[k].set_xlabel(e, fontsize = 'large')
    if e == 'year':
        ax[k].set_xticks([2015, 2016])
    if e == 'sum insured':
        ax[k].set_xlabel('S', fontsize = 'large')
        ax[k].set_xticks([0,5e5,1e6])

# data.hist(bins=100, grid = False, weights=np.zeros(N) + 1. / N, color = 'gray')
plt.tight_layout()
plt.savefig(os.path.join(path_data,r'data_marginal_dists.eps'))
plt.savefig(os.path.join(path_data,r'data_marginal_dists.png'), dpi=400)
# plt.show()
plt.close()

# print(data.corr('pearson'))
# print(data.corr('pearson').to_latex())

# sns.heatmap(data.corr('pearson'))
# plt.show()