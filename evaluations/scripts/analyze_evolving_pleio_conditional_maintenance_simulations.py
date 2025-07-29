#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load data
non_evolving_df = pd.read_csv('../data/non_evolving_pleio_conditional_decay.csv')
evolving_df = pd.read_csv('../data/evolving_pleio_conditional_decay.csv')

sns.set_style('whitegrid')

fig, ax1 = plt.subplots(1,1, figsize=(4, 4))

for index, row in evolving_df.iterrows():
    sns.lineplot(list(row), ax=ax1, legend=False, color = 'black', alpha=0.5, linewidth=2)
# plot mean dynamics
sns.lineplot(list(evolving_df.mean(axis=0)), ax=ax1, legend=False, color = 'black', linewidth=2, label = r'Evolving $r$')

for index, row in non_evolving_df.iterrows():
    sns.lineplot(list(row), ax=ax1, legend=False, color = 'tab:gray', alpha=0.5, linewidth=2)
# plot mean dynamics
sns.lineplot(list(non_evolving_df.mean(axis=0)), ax=ax1, legend=False, color = 'tab:gray', linewidth=2, label = r'Non-evolving $r$')

ax1.tick_params(axis='both', labelsize=12)
ax1.set_ylabel(r'Mean $W^C$', fontsize=12)
ax1.set_xlabel('Generations', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()