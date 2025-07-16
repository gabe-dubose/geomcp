#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from scipy.ndimage import gaussian_filter
from scipy.interpolate import UnivariateSpline
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# intiialize figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

##### ax1: parameter space surface #####

# formatting
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"Relative expression ($r$)", fontsize = 15)
ax1.set_ylabel(r"Expression frequency ($f_C$)", fontsize=15)
ax1.tick_params(axis='both', labelsize=12)


# uncomment to set up colorbar. You will need to edit this 
# to fit your code, but it has the general
# formatting, such as labels, fontsizes, and positioning
'''
cbar = fig.colorbar(contourf, ax=ax1, pad=0.03, ticks=bounds[::int(len(bounds)/n_colors)])
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(r"$\lambda_C$", fontsize=15, pad=10)
'''

##### ax2: lambda_C(f_C) lines #####

# formatting
ax2.set_xlabel(r"Expression frequency ($f_C$)", fontsize=15)
ax2.set_ylabel(r"Decay rate ($\lambda_C$)", fontsize=15)
ax2.tick_params(axis='both', labelsize=12)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])

# uncomment to set up colorbar. You will need to edit this 
# to fit your code, but it has the general
# formatting, such as labels, fontsizes, and positioning
'''
sm_b = ScalarMappable(cmap=viridis, norm=mpl.colors.Normalize(vmin=0.5, vmax=1.0))
sm_b.set_array([])
cbar_b = plt.colorbar(sm_b, ax=ax2, orientation='vertical', pad=0.03)
cbar_b.ax.tick_params(labelsize=12)
cbar_b.ax.set_title(r"$r$", fontsize=15, pad=10)
'''

# panel labels
ax1.text(-0.24, 1.15, 'A', transform=ax1.transAxes, fontsize=17, va='top', ha='left')
ax2.text(-0.32, 1.15, 'B', transform=ax2.transAxes, fontsize=17, va='top', ha='left')

plt.tight_layout()
plt.show()