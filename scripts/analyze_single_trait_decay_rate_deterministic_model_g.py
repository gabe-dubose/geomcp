#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from scipy.interpolate import UnivariateSpline
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# intiialize figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

##### ax1: parameter space surface #####

ax1.set_xlim(0, 1)
ax1.set_ylim(0, 100)
ax1.set_xlabel(r"Relative expression ($r$)", fontsize = 15)
ax1.set_ylabel(r"Gens. between expr. ($g$)", fontsize=15)
ax1.tick_params(axis='both', labelsize=12)

# uncomment to set up for colorbar for ax2. This will
# need to be edited to fit your code, but it has the general
# formatting, such as labels, fontsizes, and positioning
'''
cbar = fig.colorbar(contourf, ax=ax1, pad=0.03, ticks=bounds[::int(len(bounds)/n_colors)])
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(r"$\lambda_C$", fontsize=15, pad=10)
'''

##### ax2: lambda_C(g) lines #####

ax2.set_xlabel(r"Gens. between expr. ($g$)", fontsize=15)
ax2.set_ylabel(r"Decay rate ($\lambda_C$)", fontsize=15)
ax2.tick_params(axis='both', labelsize=12)
ax2.set_xticks([0, 25, 50, 75, 100])

# uncomment to set up for colorbar for ax2. This will
# need to be edited to fit your code, but it has the general
# formatting, such as labels, fontsizes, and positioning
'''
viridis = plt.cm.get_cmap('viridis', len(r_values))
sm_b = ScalarMappable(cmap=viridis, norm=mpl.colors.Normalize(vmin=0.5, vmax=1.0))
sm_b.set_array([])
cbar_b = plt.colorbar(sm_b, ax=ax2, orientation='vertical', pad=0.03)
cbar_b.ax.tick_params(labelsize=12)
cbar_b.ax.set_title(r"$r$", fontsize=15, pad=10)
'''

# panel labels
ax1.text(-0.26, 1.15, 'A', transform=ax1.transAxes, fontsize=17, va='top', ha='left')
ax2.text(-0.32, 1.15, 'B', transform=ax2.transAxes, fontsize=17, va='top', ha='left')