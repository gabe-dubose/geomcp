#!/usr/bin/env python3

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.colors import Normalize
from matplotlib import cm
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# intiialize figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

##### ax1: parameter space surface #####

sigma_m = 0.1      # mutational variance per generation
sigma_s = 1.0      # stabilizing selection strength
t_fit   = 50       # number of generations to iterate

r_vals = np.linspace(0.0, 1.0, 505)   # r-grid
f_vals = np.linspace(0.0, 1.0, 505)   # f_C-grid

# compute Lambda matrix
Lambda = np.zeros((len(f_vals), len(r_vals)))
for i, r in enumerate(r_vals):
    rr2 = r**2
    omr2 = (1 - r)**2
    for j, f in enumerate(f_vals):
        # skip the singular point at f=r=0 if you wish
        V = 0.0
        for _ in range(t_fit):
            S = sigma_s * (f*rr2 + (1-f)*omr2)
            V = V / (1 + 2*S*V) + sigma_m**2
        Lambda[j, i] = 0.5/t_fit * np.log(1 + 2*sigma_s*rr2*V)

cmap = cm.viridis

# Panel A: contour of λ_C
levels = np.linspace(0, np.nanmax(Lambda), 50)
cf = ax1.contourf(r_vals, f_vals, Lambda, levels=levels, cmap=cmap)
# dividing contour lines as in original
ax1.contour(r_vals, f_vals, Lambda, levels=levels[::5], colors='white', linewidths=0.8)


# Add colorbar
cbar = fig.colorbar(cf, ax=ax1)
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(r"$\lambda_C$", fontsize=15, pad=10)

# formatting
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"Relative expression ($r$)", fontsize = 15)
ax1.set_ylabel(r"Expression frequency ($f_C$)", fontsize=15)
ax1.tick_params(axis='both', labelsize=12)


##### ax2: lambda_C(f_C) lines #####

r_dense = np.linspace(0.0, 1.0, 505)  # dense r-grid for interpolation
r_sparse = np.arange(0.5, 1.01, 0.02)             # sparse r-grid for visual clarity
viridis = cm.get_cmap('viridis', len(r_sparse))   # colormap for the sparse r values

# Panel B: slices λ_C vs f_C
for k, r in enumerate(r_sparse):
    # find the matching index in the original dense grid
    i_dense = np.argmin(np.abs(r_dense - r))
    # use the sparse‐grid index for color, but pull Lambda[:, i_dense]
    ax2.plot(
        f_vals,
        Lambda[:, i_dense],
        color=viridis(k),
        lw=2.5
    )

# Add colorbar for r
sm_b = ScalarMappable(
    cmap=viridis,
    norm=mpl.colors.Normalize(vmin=0.5, vmax=1.0)
)
sm_b.set_array(r_sparse)
cbar_b = plt.colorbar(sm_b, ax=ax2, orientation='vertical', pad=0.03)
cbar_b.ax.tick_params(labelsize=12)
cbar_b.ax.set_title(r"$r$", fontsize=15, pad=10)

# formatting
ax2.set_xlabel(r"Expression frequency ($f_C$)", fontsize=15)
ax2.set_ylabel(r"Decay rate ($\lambda_C$)", fontsize=15)
ax2.tick_params(axis='both', labelsize=12)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])

# panel labels
ax1.text(-0.24, 1.15, 'A', transform=ax1.transAxes, fontsize=17, va='top', ha='left')
ax2.text(-0.32, 1.15, 'B', transform=ax2.transAxes, fontsize=17, va='top', ha='left')

plt.tight_layout()
plt.show()


##Save the figure
plt.savefig("single_trait_decay_rate_fc_variable.png", dpi=600)