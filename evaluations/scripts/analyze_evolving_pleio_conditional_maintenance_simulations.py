#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.cm as cm

# load data
non_evolving_df = pd.read_csv('../data/non_evolving_pleio_conditional_decay.csv')
evolving_df = pd.read_csv('../data/evolving_pleio_conditional_decay.csv')

sns.set_style('whitegrid')

# define colors
n_col = 3
viridis = cm.get_cmap('viridis', n_col)
colors = [viridis(i) for i in range(n_col)]

fig, ax1 = plt.subplots(1,1, figsize=(4, 4))

for index, row in evolving_df.iterrows():
    sns.lineplot(list(row), ax=ax1, legend=False, color = colors[0], alpha=0.5, linewidth=2)
# plot mean dynamics
sns.lineplot(list(evolving_df.mean(axis=0)), ax=ax1, legend=False, color = colors[0], linewidth=2, label = r'Evolving $r$')

for index, row in non_evolving_df.iterrows():
    sns.lineplot(list(row), ax=ax1, legend=False, color = colors[2], alpha=0.5, linewidth=2)
# plot mean dynamics
sns.lineplot(list(non_evolving_df.mean(axis=0)), ax=ax1, legend=False, color = colors[2], linewidth=2, label = r'Non-evolving $r$')

ax1.tick_params(axis='both', labelsize=12)
ax1.set_ylabel(r'Mean $W^C$', fontsize=12)
ax1.set_xlabel('Generations', fontsize=12)
ax1.legend(loc='upper right', fontsize=12)

plt.tight_layout()
plt.show()
