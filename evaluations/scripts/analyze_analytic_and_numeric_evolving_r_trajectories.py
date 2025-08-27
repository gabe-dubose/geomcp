#!/usr/bin/env python3
"""
panels_merged.py

Compute and plot combined Panels A, B & C:
 - Panel A: I(r) vs r for each o^A (original Panel B)
 - Panel B: dI/dr vs r for each o^A (original Panel C)
 - Panel C: High‐init pleiotropy trait dynamics under low vs high o^A (original Panel A)

Produces a single PNG with three panels (A, B, C), each labeled inside the plot. Colorbars in A & B are labeled with o^A.
"""
import os
import pandas as pd
import numpy as np
from scipy.special import erf
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

# === PARAMETER SECTION ===
# Data directory for Panel C files
data_dir = '/home/ellcharles_wang/Desktop/geomcp/data'
# Panel A/B parameters
oA_list    = np.arange(0.1, 3.0, 0.2)
sigma_s    = 1.0          # stabilizing‐selection strength
z_min, z_max = -3.0, 3.0
r_max      = 0.99
n_points   = 500
# Color settings for Panel C lines
n_col       = 3
viridis_map = cm.get_cmap('viridis', n_col)
colors      = [viridis_map(i) for i in range(n_col)]
# ===========================

def I_of_r(r, sigma_s, oA, z_min, z_max):
    s = np.sqrt(sigma_s)
    a = 1.0 - r
    if np.any(a == 0):
        raise ValueError("r must be < 1.0 to avoid divergence.")
    u_max = s * (a * z_max - oA)
    u_min = s * (a * z_min - oA)
    prefac = np.sqrt(np.pi) / (2 * s * a)
    return prefac * (erf(u_max) - erf(u_min))


def dI_dr(r, sigma_s, oA, z_min, z_max):
    s = np.sqrt(sigma_s)
    a = 1.0 - r
    if np.any(a == 0):
        raise ValueError("r must be < 1.0 to avoid divergence.")
    u_max = s * (a * z_max - oA)
    u_min = s * (a * z_min - oA)
    term1 = np.sqrt(np.pi) / (2 * s * a**2) * (erf(u_max) - erf(u_min))
    term2 = (-z_max * np.exp(-u_max**2) + z_min * np.exp(-u_min**2)) / a
    return term1 + term2


def plot_panels_merged():
    # create figure and subplots
    fig, (axA, axB, axC) = plt.subplots(
        1, 3, figsize=(15, 4),
        gridspec_kw={'width_ratios': [1, 1, 0.84]}
    )

    # --- Panel A (original Panel B) ---
    r_vals = np.linspace(0.0, r_max, n_points)
    cmapA = cm.get_cmap('viridis', len(oA_list))
    normA = Normalize(vmin=oA_list.min(), vmax=oA_list.max())
    for idx, oA in enumerate(oA_list):
        axA.plot(r_vals, I_of_r(r_vals, sigma_s, oA, z_min, z_max), color=cmapA(idx), lw=3.5)
    smA = ScalarMappable(cmap=cmapA, norm=normA)
    smA.set_array(oA_list)
    cbarA = fig.colorbar(smA, ax=axA, orientation='vertical', pad=0.03)
    cbarA.ax.set_title(r'$o^A$', fontsize=16)
    cbarA.ax.tick_params(labelsize=16)
    axA.set_xlabel('r', fontsize=20)
    axA.set_ylabel('I(r)', fontsize=20)
    axA.tick_params(labelsize=16)
    axA.text(-0.16, 1.15, 'A', transform=axA.transAxes, fontsize=22, va='top', ha='left')

    # --- Panel B (original Panel C) ---
    for idx, oA in enumerate(oA_list):
        axB.plot(r_vals, dI_dr(r_vals, sigma_s, oA, z_min, z_max), color=cmapA(idx), lw=3.5)
    axB.axhline(0, color='black', linestyle='--', linewidth=3)
    cbarB = fig.colorbar(smA, ax=axB, orientation='vertical', pad=0.03)
    cbarB.ax.set_title(r'$o^A$', fontsize=16)
    cbarB.ax.tick_params(labelsize=16)
    axB.set_xlabel('r', fontsize=20)
    axB.set_ylabel(r'$dI/dr$', fontsize=20)
    axB.tick_params(labelsize=16)
    axB.text(-0.26, 1.15, 'B', transform=axB.transAxes, fontsize=22, va='top', ha='left')

    # --- Panel C (original Panel A) ---
    df_low  = pd.read_csv(os.path.join(data_dir, 'evolving_pleio_high_init_low_oA.csv'))
    df_high = pd.read_csv(os.path.join(data_dir, 'evolving_pleio_high_init_high_oA.csv'))
    import seaborn as sns
    sns.set_style('whitegrid')
    for _, row in df_low.iterrows():
        sns.lineplot(list(row), ax=axC, legend=False, color=colors[0], alpha=0.25, linewidth=3)
    sns.lineplot(list(df_low.mean(axis=0)), ax=axC, legend=False, color=colors[0], linewidth=3, label=r'$o^A < z_{init}$')
    for _, row in df_high.iterrows():
        sns.lineplot(list(row), ax=axC, legend=False, color=colors[2], alpha=0.25, linewidth=3)
    sns.lineplot(list(df_high.mean(axis=0)), ax=axC, legend=False, color=colors[2], linewidth=3, label=r'$o^A \gg z_{init}$')
    axC.set_xlabel('Generations', fontsize=16)
    axC.set_ylabel(r'Mean $r$', fontsize=16)
    axC.tick_params(labelsize=16)
    axC.legend(loc='upper left', fontsize=14)
    axC.set_xlim(0, 50)
    axC.text(-0.22, 1.15, 'C', transform=axC.transAxes, fontsize=22, va='top', ha='left')

    plt.tight_layout()
    plt.savefig('panels_merged.png', dpi=300)
    plt.show()

if __name__ == '__main__':
    plot_panels_merged()

