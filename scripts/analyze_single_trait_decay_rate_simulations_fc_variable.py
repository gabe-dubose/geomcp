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

# read data
decay_rates = pd.read_csv('../data/decay_rates_data.csv', index_col=0)

# define color paletter
n_colors = 10
cmap = ListedColormap(sns.color_palette("viridis", n_colors))
bounds = np.linspace(decay_rates.min().min(), decay_rates.max().max(), n_colors + 1)
norm = BoundaryNorm(bounds, ncolors=n_colors)

fig, ax1 = plt.subplots(1,1, figsize = (5,4))
heatmap = sns.heatmap(decay_rates.T, cmap=cmap, norm=norm, ax=ax1, cbar_kws={"ticks": bounds})
ax1.invert_yaxis()

# clear up color bar
def sci_notation(x, pos):
    return f"{x:.1e}"    
cbar = heatmap.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(FuncFormatter(sci_notation))
cbar.set_label(r"Decay rate ($\lambda$)", fontsize=10)

# clean up axes
ax1.set_ylabel(r"$f_C$")
ax1.set_xlabel(r"$r$")

# add reference lines
ax1.axhline(y=50, color='white', linestyle='--', linewidth=1)
ax1.axvline(x=50, color='white', linestyle='--', linewidth=1)

# plot raw data: note figure must be manually closed to view subsequent figures
plt.tight_layout()
plt.show()

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 4))

# transpose data
data = decay_rates.T

# apply gaussian filter to smooth decay rate values
zz = gaussian_filter(data.values, sigma=4)

# setup for plotting
ny, nx = data.shape
x_vals = np.linspace(0, 1, nx)
y_vals = np.linspace(0, 1, ny)
X, Y = np.meshgrid(x_vals, y_vals)

# color setup
n_colors = 10
cmap = ListedColormap(sns.color_palette("viridis", n_colors))
bounds = np.linspace(0, np.max(zz), n_colors + 1)
norm = BoundaryNorm(bounds, ncolors=n_colors)

# lambda_C surface
contourf = ax1.contourf(X, Y, zz, levels=bounds, cmap=cmap, norm=norm)
contours = ax1.contour(X, Y, zz, levels=bounds[1:], colors='white', linewidths=1)
cbar = fig.colorbar(contourf, ax=ax1, pad=0.03, ticks=bounds[::int(len(bounds)/n_colors)])
cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:.1e}"))
cbar.ax.tick_params(labelsize=12)
cbar.ax.set_title(r"$\lambda_C$", fontsize=15, pad=10)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)
ax1.set_xlabel(r"Relative expression ($r$)", fontsize = 15)
ax1.set_ylabel(r"Expression frequency ($f_C$)", fontsize=15)
ax1.tick_params(axis='both', labelsize=12)

# define values
r_values = np.arange(0.5, 1.01, 0.01)
viridis = plt.cm.get_cmap('viridis', len(r_values))
dfC = y_vals[1] - y_vals[0]

# lambda_C(f_C) lines 
for i, r in enumerate(r_values):
    r_index = (np.abs(x_vals - r)).argmin()
    r_actual = x_vals[r_index]

    decay_at_r = zz[:, r_index]
    color = viridis(i)

    # Plot decay rate vs. f_C
    ax2.plot(y_vals, decay_at_r, color=color, linewidth=2.5)

ax2.set_xlabel(r"Expression frequency ($f_C$)", fontsize=15)
ax2.set_ylabel(r"Decay rate ($\lambda_C$)", fontsize=15)
ax2.tick_params(axis='both', labelsize=12)

# add colorbar
sm_b = ScalarMappable(cmap=viridis, norm=mpl.colors.Normalize(vmin=0.5, vmax=1.0))
sm_b.set_array([])
cbar_b = plt.colorbar(sm_b, ax=ax2, orientation='vertical', pad=0.03)
#cbar_b.set_label(r"Relative expression ($r$)", fontsize=15)
cbar_b.ax.tick_params(labelsize=12)
cbar_b.ax.set_title(r"$r$", fontsize=15, pad=10)

# derivatives
for i, r in enumerate(r_values):
    r_index = (np.abs(x_vals - r)).argmin()
    r_actual = x_vals[r_index]

    decay_at_r = zz[:, r_index]
    decay_gradient = np.gradient(decay_at_r, dfC)

    # Fit spline
    spline = UnivariateSpline(y_vals, decay_gradient, s=0.000075)
    y_vals_fine = np.linspace(0, 1, 200)
    pred_gradient = spline(y_vals_fine)

    color = viridis(i)
    ax3.plot(y_vals_fine, pred_gradient, color=color, linewidth=2.5)

ax3.set_xlabel(r"Expression frequency ($f_C$)", fontsize=15)
ax3.set_ylabel(r"$d \lambda_C$ / $d f_C$", fontsize=15)
ax3.set_ylim(-0.015, 0)
ax3.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.1e}"))
ax3.tick_params(axis='both', labelsize=12)

# add colorbar
sm_c = ScalarMappable(cmap=viridis, norm=mpl.colors.Normalize(vmin=0.5, vmax=1.0))
sm_c.set_array([])
cbar_c = plt.colorbar(sm_c, ax=ax3, orientation='vertical', pad=0.03)
#cbar_c.set_label(r"Relative expression ($r$)", fontsize=15)
cbar_c.ax.tick_params(labelsize=12)
cbar_c.ax.set_title(r"$r$", fontsize=15, pad=10)

# Panel labels
ax1.text(-0.22, 1.15, 'A', transform=ax1.transAxes, fontsize=17, va='top', ha='left')
ax2.text(-0.28, 1.15, 'B', transform=ax2.transAxes, fontsize=17, va='top', ha='left')
ax3.text(-0.375, 1.15, 'C', transform=ax3.transAxes, fontsize=17, va='top', ha='left')

ax2.set_xticks([0, 0.25, 0.5, 0.75, 1])
ax3.set_xticks([0, 0.25, 0.5, 0.75, 1])

plt.tight_layout()
plt.show()


