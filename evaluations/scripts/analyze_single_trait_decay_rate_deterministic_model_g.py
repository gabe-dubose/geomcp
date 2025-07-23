#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
import warnings
warnings.filterwarnings('ignore')

# initialize figure
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

##### ax1: parameter space surface #####

sigma_m = 0.1      # mutational variance per generation
sigma_s = 1.0      # stabilizing selection strength
t_fit   = 50       # number of generations to iterate

# dense grids for r and f (f_C)
r_vals = np.linspace(0.0, 1.0, 505)   # r-grid
f_vals = np.linspace(0.0, 1.0, 10001)   # fine f_C-grid to ensure proper resolution of g

# compute Lambda matrix
Lambda = np.zeros((len(f_vals), len(r_vals)))
for i, r in enumerate(r_vals):
    rr2 = r**2
    omr2 = (1 - r)**2
    for j, f in enumerate(f_vals):
        V = 0.0
        for _ in range(t_fit):
            S = sigma_s * (f * rr2 + (1 - f) * omr2)
            V = V / (1 + 2 * S * V) + sigma_m**2
        Lambda[j, i] = 0.5 / t_fit * np.log(1 + 2 * sigma_s * rr2 * V)

# derive g = 1/f (handle f=0)
g_vals = np.empty_like(f_vals)
g_vals[0] = np.nan
f_nonzero = f_vals[1:]
g_vals[1:] = 1 / f_nonzero

# build a grid of integer g from 1 to 100
g_int = np.arange(2, 101)
# for each integer g, find closest f index
idx_f = [np.argmin(np.abs(f_vals - 1.0 / g)) for g in g_int]
# extract corresponding Lambda rows
Lambda_g = Lambda[idx_f, :]

# Panel A: contour of lambda_C over (r, g_int)
levels = np.linspace(0, np.nanmax(Lambda_g), 50)
contourf = ax1.contourf(
    r_vals,
    g_int,
    Lambda_g,
    levels=levels,
    cmap='viridis'
)
ax1.contour(
    r_vals,
    g_int,
    Lambda_g,
    levels=levels[::5],
    colors='white',
    linewidths=0.8
)

# colorbar for ax1
cbar = fig.colorbar(contourf, ax=ax1, pad=0.03,
                    ticks=levels[::int(len(levels) / 10)])
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(r"$\lambda_C$", fontsize=15, pad=10)

# formatting axes for ax1
ax1.set_xlim(0, 1)
ax1.set_ylim(2, 100)
ax1.set_xlabel(r"Relative expression ($r$)", fontsize=15)
ax1.set_ylabel(r"Gens. between expr. ($g$)", fontsize=15)
ax1.tick_params(axis='both', labelsize=12)

##### ax2: lambda_C(g) lines #####
# plot each r-slice against g_int
r_dense = r_vals.copy()
r_sparse = np.arange(0.5, 1.01, 0.02)
viridis = plt.cm.get_cmap('viridis', len(r_sparse))
for k, r in enumerate(r_sparse):
    i_dense = np.argmin(np.abs(r_dense - r))
    ax2.plot(
        g_int,
        Lambda_g[:, i_dense],
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

# formatting axes for ax2
ax2.set_xlim(1, 100)
ax2.set_xlabel(r"Gens. between expr. ($g$)", fontsize=15)
ax2.set_ylabel(r"Decay rate ($\lambda_C$)", fontsize=15)
ax2.tick_params(axis='both', labelsize=12)
ax2.set_xticks([1, 25, 50, 75, 100])

# panel labels
ax1.text(-0.26, 1.15, 'A', transform=ax1.transAxes, fontsize=17, va='top', ha='left')
ax2.text(-0.32, 1.15, 'B', transform=ax2.transAxes, fontsize=17, va='top', ha='left')

plt.tight_layout()
plt.show()

# Save the figure
# plt.savefig("single_trait_decay_rate_g_variable.png", dpi=600)





