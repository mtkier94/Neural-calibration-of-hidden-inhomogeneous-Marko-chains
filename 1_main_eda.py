'''
Perform Exploratory Data Analysis
'''

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

data.hist(bins=100, grid = False, weights=np.zeros(N) + 1. / N)
plt.tight_layout()
plt.show()

print(data.corr('pearson'))
print(data.corr('pearson').to_latex())

sns.heatmap(data.corr('pearson'))
plt.show()