#!/usr/bin/env python3

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.ticker import FuncFormatter
from matplotlib.cm import ScalarMappable
import matplotlib as mpl
from scipy.ndimage import gaussian_filter
from scipy.ndimage import gaussian_filter1d
import matplotlib.lines as mlines
import warnings
warnings.filterwarnings('ignore')

# =====================
# initialize figure
# =====================
fig, [ax1, ax2] = plt.subplots(1, 2, figsize=(9, 4))

# =====================
# parameters
# =====================
sigma_m = 0.1      # mutational variance per generation
sigma_s = 1.0      # stabilizing selection strength
t_fit   = 50       # number of generations to iterate

# grids for r and g
r_vals = np.linspace(0.0, 1.0, 505)   # r-grid
g_int = np.arange(2, 101)             # integer g values
f_targets = 1.0 / g_int               # f = 1/g

# =====================
# compute Lambda at (r, f=1/g)
# =====================
Lambda_g = np.zeros((len(g_int), len(r_vals)))
for i, r in enumerate(r_vals):
    rr2 = r**2
    omr2 = (1 - r)**2
    for j, f in enumerate(f_targets):
        V = 0.0
        for _ in range(t_fit):
            S = sigma_s * (f * rr2 + (1 - f) * omr2)
            V = V / (1 + 2 * S * V) + sigma_m**2
        Lambda_g[j, i] = 0.5 / t_fit * np.log(1 + 2 * sigma_s * rr2 * V)

# =====================
# Panel A: contour of Lambda_C over (r, g)
# =====================
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

# =====================
# Panel B: Lambda_C(g) lines
# =====================
r_sparse = np.arange(0.5, 1.01, 0.1)
viridis = plt.cm.get_cmap('viridis', len(r_sparse))

# --- Analytical curves ---
for k, r in enumerate(r_sparse):
    i_dense = np.argmin(np.abs(r_vals - r))
    ax2.plot(
        g_int,
        Lambda_g[:, i_dense],
        color=viridis(k),
        lw=2.5
    )

# --- Numerical curves ---
decay_rates = pd.read_csv('../data/decay_rates_data_temporal.csv', index_col=0)
data = decay_rates.T

# 2D Gaussian filter across the matrix
zz = gaussian_filter(data.values, sigma=6)

ny, nx = data.shape
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(1, 100, ny)  # match g scale

for k, r in enumerate(r_sparse):
    r_index = (np.abs(x_vals - r)).argmin()
    decay_at_r = zz[:, r_index]

    # =====================
    # 1D Gaussian smoothing along g
    # =====================
    decay_at_r_smooth = gaussian_filter1d(decay_at_r, sigma=4)

    ax2.plot(
        y_vals,
        decay_at_r_smooth,
        color=viridis(k),
        lw=2.0,
        linestyle="--"
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

# =====================
# panel labels
# =====================
ax1.text(-0.26, 1.15, 'A', transform=ax1.transAxes,
         fontsize=17, va='top', ha='left')
ax2.text(-0.32, 1.15, 'B', transform=ax2.transAxes,
         fontsize=17, va='top', ha='left')

# add panel b legend
analytic_handle = mlines.Line2D([], [], color="tab:gray", linestyle="-", lw=2, label="Analytical")
numeric_handle  = mlines.Line2D([], [], color="tab:gray", linestyle="--", lw=2, label="Numerical")

ax2.legend(
    handles=[analytic_handle, numeric_handle],
    loc="upper left",
    fontsize=12,
    frameon=False
)

plt.tight_layout()
plt.show()

# Save the figure
#plt.savefig("single_trait_decay_rate_g_variable.png", dpi=600)
