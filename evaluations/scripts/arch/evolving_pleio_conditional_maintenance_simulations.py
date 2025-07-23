#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# model function
def evolve_r(n_traits, z_0, r_init, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations):
    
    # initialize vectors to track dynamics
    mean_W_C = []
    mean_W_A = []
    mean_P_C = []
    mean_P_A = []
    mean_r = []
    mean_z = []

    # initialize population
    population = {}
    W_C_0 = []
    W_A_0 = []
    P_C_0 = []
    P_A_0 = []
    r_0 = []
    z_i = []

    for i in range(population_size):
        individual_z = z_0 + np.random.normal(mu, sigma_m, n_traits)
        individual_r = r_init.copy()
        P_C = individual_z * individual_r
        P_A = individual_z * (1 - individual_r)
        dist_C = np.sum((P_C - opt_C) ** 2)
        dist_A = np.sum((P_A - opt_A) ** 2)
        W_C = np.exp(-sigma_s * dist_C)
        W_A = np.exp(-sigma_s * dist_A)
        W_T = W_C * f_C + W_A * f_A
        W_ = [W_T, W_C, W_A]
        population[i] = [individual_z, individual_r, W_]

        W_C_0.append(W_C)
        W_A_0.append(W_A)
        P_C_0.append(P_C)
        P_A_0.append(P_A)
        r_0.append(individual_r)
        z_i.append(individual_z)

    # track initial means
    mean_W_C.append(np.mean(W_C_0))
    mean_W_A.append(np.mean(W_A_0))
    mean_P_C.append(np.mean(P_C_0, axis=0))
    mean_P_A.append(np.mean(P_A_0, axis=0))
    mean_r.append(np.mean(r_0, axis=0))
    mean_z.append(np.mean(z_i, axis=0))

    # evolution loop
    for generation in range(generations):
        fitness_vals = [population[i][2][0] for i in population]
        total_fitness = sum(fitness_vals)
        rel_fitness = [f / total_fitness for f in fitness_vals]
        parents = np.random.choice(list(population.keys()), size=population_size, p=rel_fitness)
        
        next_gen = {}
        W_C_pop, W_A_pop, P_C_pop, P_A_pop, r_pop, z_pop = [], [], [], [], [], []

        for i, p in enumerate(parents):
            z_parent, r_parent, _ = population[p]
            delta_z = np.random.normal(mu, sigma_m, n_traits)
            delta_r = np.random.normal(0, sigma_r, n_traits)
            new_z = z_parent + delta_z
            new_r = np.clip(r_parent + delta_r, 0, 1)

            new_P_C = new_z * new_r
            new_P_A = new_z * (1 - new_r)
            dist_C = np.sum((new_P_C - opt_C) ** 2)
            dist_A = np.sum((new_P_A - opt_A) ** 2)
            W_C = np.exp(-sigma_s * dist_C)
            W_A = np.exp(-sigma_s * dist_A)
            W_T = W_C * f_C + W_A * f_A
            W_ = [W_T, W_C, W_A]

            next_gen[i] = [new_z, new_r, W_]

            W_C_pop.append(W_C)
            W_A_pop.append(W_A)
            P_C_pop.append(new_P_C)
            P_A_pop.append(new_P_A)
            r_pop.append(new_r)
            z_pop.append(new_z)

        mean_W_C.append(np.mean(W_C_pop))
        mean_W_A.append(np.mean(W_A_pop))
        mean_P_C.append(np.mean(P_C_pop, axis=0))
        mean_P_A.append(np.mean(P_A_pop, axis=0))
        mean_r.append(np.mean(r_pop, axis=0))
        mean_z.append(np.mean(z_pop, axis=0))

        population = next_gen

    dynamics = {
        'mean_W_C': mean_W_C,
        'mean_W_A': mean_W_A,
        'mean_P_C': mean_P_C,
        'mean_P_A': mean_P_A,
        'mean_r': mean_r, # shape: (generations+1, n_traits)
        'mean_z' : mean_z # shape: (generations+1, n_traits)
    }

    return dynamics

# PARAMETERS
# PHENOTYPE
n_traits = 20 # --- number of traits
z_0 = np.ones(n_traits) # --- initial trait values
# MUTATIONS
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 0.1 # --- standard deviation in mutation effect on z
# POPULATION/FITNESS
population_size = 1000 # --- population size
sigma_s = 1 # --- width of fitness peak 
f_C = 0 # --- frequency/probability of conditional
f_A = 1 # --- frequency/probability of alternative
generations = 500

# define variable r distributions
# highly pleiotropic
alpha = 20 # --- trait bias towards conditional phenotype
beta = 20 # --- trait bias towards alternative phenotype
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, n_traits)
r_high = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution
opt_C_high = r_high * z_0 # --- optimal phenotype for conditional 
opt_A_high = (1-r_high) * z_0 # --- optimal phenotype for alternative

# less pleiotropic
alpha = 2 # --- trait bias towards conditional phenotype
beta = 2 # --- trait bias towards alternative phenotype
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, n_traits)
r_low = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution
opt_C_low = r_low * z_0 # --- optimal phenotype for conditional 
opt_A_low = (1-r_low) * z_0 # --- optimal phenotype for alternative

# no pleiotropy
prop_cond = int(n_traits * 0.5)
prop_alt = n_traits - prop_cond
r_no = np.array([1] * prop_cond + [0] * prop_alt)
opt_C_no = r_no * z_0 # --- optimal phenotype for conditional 
opt_A_no = (1-r_no) * z_0 # --- optimal phenotype for alternative

# function to iteratively run simulations
def iterate_simulations(n_runs, sigma_r):
    for i in range(n_runs):

        if i == 0:
            # run and initialize dataframes
            r_no_history = evolve_r(n_traits, z_0, r_no, opt_C_no, opt_A_no, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            r_no_history_df = pd.DataFrame(r_no_history['mean_W_C'])
            r_no_r_df = pd.DataFrame(list(r_no_history['mean_r'][-1]))
            z_no_r_df = pd.DataFrame(list(r_no_history['mean_z'][-1]))
            
            # run and initialize dataframes
            r_low_history = evolve_r(n_traits, z_0, r_low, opt_C_low, opt_A_low, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            r_low_history_df = pd.DataFrame(r_low_history['mean_W_C'])
            r_low_r_df = pd.DataFrame(list(r_low_history['mean_r'][-1]))
            z_low_r_df = pd.DataFrame(list(r_low_history['mean_z'][-1]))
            
            # run and initialize dataframes
            r_high_history = evolve_r(n_traits, z_0, r_high, opt_C_high, opt_A_high, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            r_high_history_df = pd.DataFrame(r_high_history['mean_W_C'])
            r_high_r_df = pd.DataFrame(list(r_high_history['mean_r'][-1]))
            z_high_r_df = pd.DataFrame(list(r_high_history['mean_z'][-1]))
            
        else:
            # run
            r_no_history = evolve_r(n_traits, z_0, r_no, opt_C_no, opt_A_no, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            r_low_history = evolve_r(n_traits, z_0, r_low, opt_C_low, opt_A_low, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            r_high_history = evolve_r(n_traits, z_0, r_high, opt_C_high, opt_A_high, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)
            
            r_no_history_df[f"run_{i}"] = r_no_history['mean_W_C']
            r_low_history_df[f"run_{i}"] = r_low_history['mean_W_C']
            r_high_history_df[f"run_{i}"] = r_high_history['mean_W_C']

            r_no_r_df[f"run_{i}"] = list(r_no_history['mean_r'][-1])
            r_low_r_df[f"run_{i}"] = list(r_low_history['mean_r'][-1])
            r_high_r_df[f"run_{i}"] = list(r_high_history['mean_r'][-1])

            z_no_r_df[f"run_{i}"] = list(r_no_history['mean_z'][-1])
            z_low_r_df[f"run_{i}"] = list(r_low_history['mean_z'][-1])
            z_high_r_df[f"run_{i}"] = list(r_high_history['mean_z'][-1])
            

    return {'r_no' : r_no_history_df, 
           'r_low' : r_low_history_df, 
           'r_high' : r_high_history_df,
           'r_no_r' : r_no_r_df,
           'r_low_r': r_low_r_df,
           'r_high_r' : r_high_r_df,
           'z_no_r' : z_no_r_df,
           'z_low_r' : z_low_r_df,
           'z_high_r' : z_high_r_df}

results_sigma_r_001 = iterate_simulations(25, 0.01)
results_sigma_r_0 = iterate_simulations(25, 0.0)

sns.set_style("white")

# define colors
n_col = 3
viridis = cm.get_cmap('viridis', n_col)
colors = [viridis(i) for i in range(n_col)]

# initialize plot
fig, [[ax1, ax2], [ax3, ax4]] = plt.subplots(2, 2, figsize=(6,5))

# --- sigma_r = 0.01 --- #
# plot dynamics
# no pleio
for column in results_sigma_r_001['r_no'].columns:
    sns.lineplot(results_sigma_r_001['r_no'][column].values, ax=ax1, linestyle = '-', color=colors[2], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_001['r_no'].mean(axis=1).to_list(), ax=ax1, linestyle = '-', color=colors[2], linewidth=2, label = 'No pleiotropy', zorder=2, legend=False)

# low pleio
for column in results_sigma_r_001['r_low'].columns:
    sns.lineplot(results_sigma_r_001['r_low'][column].values, ax=ax1, linestyle = '-', color=colors[1], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_001['r_low'].mean(axis=1).to_list(), ax=ax1, linestyle = '-', color=colors[1], linewidth=2, label = 'Low pleiotropy', zorder=2, legend=False)

# high high
for column in results_sigma_r_001['r_high'].columns:
    sns.lineplot(results_sigma_r_001['r_high'][column].values, ax=ax1, linestyle = '-', color=colors[0], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_001['r_high'].mean(axis=1).to_list(), ax=ax1, linestyle = '-', color=colors[0], linewidth=2, label = 'High pleiotropy', zorder=2, legend=False)

# --- sigma_r = 0.0 --- #
# plot dynamics
# no pleio
for column in results_sigma_r_0['r_no'].columns:
    sns.lineplot(results_sigma_r_0['r_no'][column].values, ax=ax2, linestyle = '-', color=colors[2], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_0['r_no'].mean(axis=1).to_list(), ax=ax2, linestyle = '-', color=colors[2], linewidth=2, label = 'No pleiotropy', zorder=2, legend=False)

# low pleio
for column in results_sigma_r_0['r_low'].columns:
    sns.lineplot(results_sigma_r_0['r_low'][column].values, ax=ax2, linestyle = '-', color=colors[1], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_0['r_low'].mean(axis=1).to_list(), ax=ax2, linestyle = '-', color=colors[1], linewidth=2, label = 'Low pleiotropy', zorder=2, legend=False)

# high high
for column in results_sigma_r_0['r_high'].columns:
    sns.lineplot(results_sigma_r_0['r_high'][column].values, ax=ax2, linestyle = '-', color=colors[0], alpha=0.1, zorder=1, legend=False)
# plot mean
sns.lineplot(results_sigma_r_0['r_high'].mean(axis=1).to_list(), ax=ax2, linestyle = '-', color=colors[0], linewidth=2, label = 'High pleiotropy', zorder=2, legend=False)


ax1.set_ylabel(r'Mean $W^C$', fontsize=12)
ax1.set_ylim(0,1)
ax2.set_ylim(0,1)

# plot variances
# sigma_r = 0.01
sns.lineplot(results_sigma_r_001['r_no'].var(axis=1).to_list(), ax=ax3, linestyle = '-', color=colors[2], linewidth=2, label = 'No pleiotropy', legend=False)
sns.lineplot(results_sigma_r_001['r_low'].var(axis=1).to_list(), ax=ax3, linestyle = '-', color=colors[1], linewidth=2, label = 'Low pleiotropy', legend=False)
sns.lineplot(results_sigma_r_001['r_high'].var(axis=1).to_list(), ax=ax3, linestyle = '-', color=colors[0], linewidth=2, label = 'High pleiotropy', legend=False)

# sigma_r = 0.0
sns.lineplot(results_sigma_r_0['r_no'].var(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[2], linewidth=2, label = 'No pleiotropy', legend=False)
sns.lineplot(results_sigma_r_0['r_low'].var(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[1], linewidth=2, label = 'Low pleiotropy', legend=False)
sns.lineplot(results_sigma_r_0['r_high'].var(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[0], linewidth=2, label = 'High pleiotropy', legend=False)

ax3.set_ylabel(r'Var($W^C$) across simulations', fontsize=12)
ax3.set_xlabel('Generation', fontsize=12)
ax4.set_xlabel('Generation', fontsize=12)

ax3.set_ylim(0, 0.039)
ax4.set_ylim(0, 0.039)

# add text labels
ax1.text(325, 0.9, r'Evolving $r$', fontsize=12)
ax2.text(235, 0.9, r'Non-Evolving $r$', fontsize=12)
ax3.text(325, 0.035, r'Evolving $r$', fontsize=12)
ax4.text(235, 0.035, r'Non-evolving $r$', fontsize=12)

# add legend
handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels,
           loc='upper right',
           bbox_to_anchor=(1.3, 0.975),
          prop={'size': 12})

# add labels
fig.text(0.02, 0.96, 'A', fontsize=17)
fig.text(0.02, 0.55, 'B', fontsize=17)

# adjust text sizes
ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
ax3.tick_params(axis='both', labelsize=12)
ax4.tick_params(axis='both', labelsize=12)


plt.tight_layout()
plt.show()

sns.set_style("white")

# define colors
n_col = 3
viridis = cm.get_cmap('viridis', n_col)
colors = [viridis(i) for i in range(n_col)]

# initialize plot
fig, [[ax1,ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(2, 3, figsize=(8,5))

# plot r distributions
sns.kdeplot(r_no, ax=ax1, color = 'black', linestyle = '--', zorder=2)
for column in results_sigma_r_001['r_no_r'].columns:
    sns.kdeplot(results_sigma_r_001['r_no_r'][column].values, ax=ax1, linestyle = '-', color='tab:gray', alpha=0.75, zorder=1)

sns.kdeplot(r_low, ax=ax2, color = 'black', linestyle = '--', zorder=2)
for column in results_sigma_r_001['r_low_r'].columns:
    sns.kdeplot(results_sigma_r_001['r_low_r'][column].values, ax=ax2, linestyle = '-', color='tab:gray', alpha=0.75, zorder=1)

sns.kdeplot(r_high, ax=ax3, color = 'black', linestyle = '--', zorder=2)
for column in results_sigma_r_001['r_high_r'].columns:
    sns.kdeplot(results_sigma_r_001['r_high_r'][column].values, ax=ax3, linestyle = '-', color='tab:gray', alpha=0.75, zorder=1)

ax1.set_xlim(0,1)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)

# plot fitness surfaces
# function to get fitness surface
def fitness_surface(opt_A):
    sigma_s = 1
    r_values = np.arange(0, 1, 0.01)
    z_values = np.arange(-3, 3.0, 0.06)
    fitness_matrix = np.zeros((len(r_values), len(z_values)))
    
    for i, r in enumerate(r_values):
        for j, z in enumerate(z_values):
            z_A = (1 - r) * z
            fitness = np.exp(-((z_A - opt_A) ** 2) / (2 * sigma_s ** 2))
            fitness_matrix[i, j] = fitness
            
    fitness_df = pd.DataFrame(fitness_matrix, index=r_values, columns=z_values)
    fitness_df.index.name = 'r'
    fitness_df.columns.name = 'z'

    return fitness_df

opt_A_01_surface = fitness_surface(0.1)
opt_A_05_surface = fitness_surface(0.5)
opt_A_09_surface = fitness_surface(0.9)

ticks = [0, 0.25, 0.5, 0.75, 1]
sns.heatmap(opt_A_01_surface, cmap="viridis", vmin=0, vmax=1, cbar=False, ax=ax4)
# format x and y ticks
ax4.set_xticklabels([f"{opt_A_01_surface.columns[int(tick)]:.1f}" for tick in ax4.get_xticks()])
ax4.set_yticklabels([f"{opt_A_01_surface.index[int(tick)]:.1f}" for tick in ax4.get_yticks()])
ax4.set_title(r'$o^A$ = 0.1')
ax4.set_ylabel(r'$r$', fontsize=12)
ax4.set_xlabel(r'$z$', fontsize=12)
ax4.invert_yaxis()

# Plot heatmaps
sns.heatmap(opt_A_05_surface, cmap="viridis", vmin=0, vmax=1, cbar=False, ax=ax5)
# format x and y ticks
ax5.set_xticklabels([f"{opt_A_05_surface.columns[int(tick)]:.1f}" for tick in ax5.get_xticks()])
ax5.set_yticklabels([f"{opt_A_05_surface.index[int(tick)]:.1f}" for tick in ax5.get_yticks()])
ax5.set_title(r'$o^A$ = 0.5')
ax5.set_ylabel(r'$r$', fontsize=12)
ax5.set_xlabel(r'$z$', fontsize=12)
ax5.invert_yaxis()

# Plot heatmaps
heatmap = sns.heatmap(opt_A_09_surface, cmap="viridis", vmin=0, vmax=1, cbar=False, ax=ax6)
# format x and y ticks
ax6.set_xticklabels([f"{opt_A_09_surface.columns[int(tick)]:.1f}" for tick in ax6.get_xticks()])
ax6.set_yticklabels([f"{opt_A_09_surface.index[int(tick)]:.1f}" for tick in ax6.get_yticks()])
ax6.set_title(r'$o^A$ = 0.9')
ax6.set_ylabel(r'$r$', fontsize=12)
ax6.set_xlabel(r'$z$', fontsize=12)
ax6.invert_yaxis()

# add color bar
cbar_ax = fig.add_axes([0.975, 0.14, 0.01, 0.325])  # [left, bottom, width, height]
norm = plt.Normalize(vmin=0, vmax=1)
sm = plt.cm.ScalarMappable(cmap="viridis", norm=norm)
sm.set_array([])
cbar = fig.colorbar(sm, cax=cbar_ax, ticks=ticks)
cbar.set_label(r"$W^A$", fontsize=12)

ax1.tick_params(axis='both', labelsize=12)
ax2.tick_params(axis='both', labelsize=12)
ax3.tick_params(axis='both', labelsize=12)
ax4.tick_params(axis='both', labelsize=10)
ax5.tick_params(axis='both', labelsize=10)
ax6.tick_params(axis='both', labelsize=10)

ax1.set_ylabel('Density', fontsize=12)
ax2.set_ylabel('Density', fontsize=12)
ax3.set_ylabel('Density', fontsize=12)
ax1.set_xlabel(r'$r$', fontsize=12)
ax2.set_xlabel(r'$r$', fontsize=12)
ax3.set_xlabel(r'$r$', fontsize=12)

fig.text(0.01, 0.96, 'A', fontsize=17)
fig.text(0.01, 0.52, 'B', fontsize=17)

plt.tight_layout()
plt.show()