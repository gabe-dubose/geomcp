#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations):
    
    # initialize vectors to track dynamics
    mean_W_C = []
    mean_W_A = []
    mean_P_C = []
    mean_P_A = []
    
    # initialize a population with some standing variation
    population = {}
    # initialize vectors to partition fitness and phenotype components
    W_C_0 = []
    W_A_0 = []
    P_C_0 = []
    P_A_0 = []
    
    for i in range(population_size):
        individual_z = np.ones(n_traits) + np.random.normal(0, sigma_m, n_traits)
        # calculate initial phenotypes
        P_C = individual_z * r
        P_A = individual_z * (1-r)
        # calculate distances from optimum
        dist_C = np.sum((P_C - opt_C) ** 2)
        dist_A = np.sum((P_A - opt_A) ** 2)
        # calculate fitness components
        W_C = np.exp(-sigma_s * dist_C)
        W_A = np.exp(-sigma_s * dist_A)
        # calculate fitness and define fitness vector
        W_T = W_C * f_C + W_A * f_A
        W_ = [W_T, W_C, W_A]
        population[i] = [individual_z, W_]
        # add fitness and phenotype components
        W_C_0.append(W_C)
        W_A_0.append(W_A)
        P_C_0.append(P_C)
        P_A_0.append(P_A)

    # calculate and update means
    mean_W_C.append(sum(W_C_0) / len(W_C_0))
    mean_W_A.append(sum(W_A_0) / len(W_A_0))
    mean_P_C.append(sum(P_C_0) / len(P_C_0))
    mean_P_A.append(sum(P_A_0) / len(P_A_0))
    
    # run through time 
    for generation in range(generations):        
        # decide which individuals get to reproduce based on relative fitness
        individual_fitness_values = []
        for individual in population:
            total_fitness = population[individual][1][0]
            individual_fitness_values.append(total_fitness)
        # calculate relative fitness
        total_fitness = sum(individual_fitness_values)
        relative_fitnesses = [fit / total_fitness for fit in individual_fitness_values]
        # pick individuals to reproduce
        parentals = np.random.choice(list(population.keys()), size=population_size, p=relative_fitnesses)

        # create next generation
        # initialize dictionary to hold next generation
        next_generation = {}
        # initialize vectors to partition fitness components
        W_C_pop = []
        W_A_pop = []
        P_C_pop = []
        P_A_pop = []
        
        # for each individual in the population
        for i, parent in enumerate(parentals):
            # get parental traits
            parental_z = population[parent][0]
            # impose mutation
            delta_z = np.random.normal(0, sigma_m, n_traits)
            new_z = parental_z + delta_z
            # calculate phenotypes
            new_P_C = new_z * r
            new_P_A = new_z * (1-r)
            # calculate distances from optimum
            new_dist_C = np.sum((new_P_C - opt_C) ** 2)
            new_dist_A = np.sum((new_P_A - opt_A) ** 2)
            # calculate fitness components
            new_W_C = np.exp(-sigma_s * new_dist_C)
            new_W_A = np.exp(-sigma_s * new_dist_A)
            # calculate fitness and define fitness vector
            new_W_T = new_W_C * f_C + new_W_A * f_A
            new_W_ = [new_W_T, new_W_C, new_W_A]
            # add to next generation
            next_generation[i] = [new_z, new_W_]
            # update fitness and phenotype components
            W_C_pop.append(new_W_C)
            W_A_pop.append(new_W_A)
            P_C_pop.append(new_P_C)
            P_A_pop.append(new_P_A)
            
        # calculate and update means
        mean_W_C.append(sum(W_C_pop) / len(W_C_pop))
        mean_W_A.append(sum(W_A_pop) / len(W_A_pop))
        mean_P_C.append(sum(P_C_pop) / len(P_C_pop))
        mean_P_A.append(sum(P_A_pop) / len(P_A_pop))
        
        # update population
        population = next_generation

    # define object to return
    dynamics = {'mean_W_C' : mean_W_C, 'mean_W_A' : mean_W_A, 'mean_P_C' : mean_P_C, 'mean_P_A' : mean_P_A}
    return dynamics

# FIXED PARAMETERS
n_traits = 2 # --- number of traits
z_0 = np.ones(n_traits) # --- initial trait values
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 1 # --- standard deviation in affect
# POPULATION/FITNESS
population_size = 100 # --- population size
sigma_s = 1 # --- width of fitness peak 
f_C = 1 # --- frequency/probability of conditional
f_A = 1 # --- frequency/probability of alternative
generations = 1000

sns.set_style("white")
# define colors
n_col = 3
viridis = cm.get_cmap('viridis', n_col)
colors = [viridis(i) for i in range(n_col)]

fig, [ax1, ax2, ax3] = plt.subplots(1,3,figsize=(9, 3), sharex=True, sharey=True)

######################################
# --- biased towards alternative --- #
######################################
r = np.array([0.2, 0.2]) # --- r distribution
opt_C = r * z_0 # --- optimal phenotype for conditional 
opt_A = (1-r) * z_0 # --- optimal phenotype for alternative

# run simulations
history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)

# get conditional data
P_C_data = []
for coords in history['mean_P_C']:
    x = coords[0]
    y = coords[1]
    P_C_data.append([x, y])
P_C_df = pd.DataFrame(data=P_C_data, columns=['x', 'y'])

# get alternative data
P_A_data = []
for coords in history['mean_P_A']:
    x = coords[0]
    y = coords[1]
    P_A_data.append([x, y])
P_A_df = pd.DataFrame(data=P_A_data, columns=['x', 'y'])

# Conditional
sns.scatterplot(x='x', y='y', data=P_C_df, edgecolor=None, color=colors[0], s=10, zorder=1, ax=ax1, alpha=0.75)
sns.lineplot(x='x', y='y', data=P_C_df, color=colors[0], zorder=2, ax=ax1, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_C[0]], y=[opt_C[1]], edgecolor='white', color=colors[0], marker='o', s = 100, zorder=3, ax=ax1)

# Alternative
sns.scatterplot(x='x', y='y', data=P_A_df, edgecolor=None, color=colors[2], zorder=1, ax=ax1, s=10, alpha=0.75)
sns.lineplot(x='x', y='y', data=P_A_df, color=colors[2], zorder=2, ax=ax1, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_A[0]], y=[opt_A[1]], edgecolor='black', color=colors[2], marker='o', s = 100, zorder=3, ax=ax1)

# add r labels
ax1.text(0.05, 0.95, r"$r_{z_{1}} = 0.2$", transform=ax1.transAxes, ha='left', va='top', fontsize=10)
ax1.text(0.05, 0.85, r"$r_{z_{2}} = 0.2$", transform=ax1.transAxes, ha='left', va='top', fontsize=10)


########################
# --- similar bias --- #
########################

# biased towards alternative
r = np.array([0.4, 0.6]) # --- r distribution
opt_C = r * z_0 # --- optimal phenotype for conditional 
opt_A = (1-r) * z_0 # --- optimal phenotype for alternative

# run simulations
history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)

# get conditional data
P_C_data = []
for coords in history['mean_P_C']:
    x = coords[0]
    y = coords[1]
    P_C_data.append([x, y])
P_C_df = pd.DataFrame(data=P_C_data, columns=['x', 'y'])

# get alternative data
P_A_data = []
for coords in history['mean_P_A']:
    x = coords[0]
    y = coords[1]
    P_A_data.append([x, y])
P_A_df = pd.DataFrame(data=P_A_data, columns=['x', 'y'])

# Conditional
sns.scatterplot(x='x', y='y', data=P_C_df, edgecolor=None, color=colors[0], s=10, zorder=1, ax=ax2, alpha=0.75)
sns.lineplot(x='x', y='y', data=P_C_df, color=colors[0], zorder=2, ax=ax2, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_C[0]], y=[opt_C[1]], edgecolor='white', color=colors[0], marker='o', s = 100, zorder=3, ax=ax2)

# Alternative
sns.scatterplot(x='x', y='y', data=P_A_df, edgecolor=None, color=colors[2], zorder=1, ax=ax2, s=10, alpha=0.75)
sns.lineplot(x='x', y='y', data=P_A_df, color=colors[2], zorder=2, ax=ax2, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_A[0]], y=[opt_A[1]], edgecolor='black', color=colors[2], marker='o', s = 100, zorder=3, ax=ax2)

# add r labels
ax2.text(0.05, 0.95, r"$r_{z_{1}} = 0.4$", transform=ax2.transAxes, ha='left', va='top', fontsize=10)
ax2.text(0.05, 0.85, r"$r_{z_{2}} = 0.6$", transform=ax2.transAxes, ha='left', va='top', fontsize=10)

######################################
# --- biased towards conditional --- #
######################################
# biased towards alternative
r = np.array([0.8, 0.8]) # --- r distribution
opt_C = r * z_0 # --- optimal phenotype for conditional 
opt_A = (1-r) * z_0 # --- optimal phenotype for alternative

# run simulations
history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)

# get conditional data
P_C_data = []
for coords in history['mean_P_C']:
    x = coords[0]
    y = coords[1]
    P_C_data.append([x, y])
P_C_df = pd.DataFrame(data=P_C_data, columns=['x', 'y'])

# get alternative data
P_A_data = []
for coords in history['mean_P_A']:
    x = coords[0]
    y = coords[1]
    P_A_data.append([x, y])
P_A_df = pd.DataFrame(data=P_A_data, columns=['x', 'y'])

# Conditional
sns.scatterplot(x='x', y='y', data=P_C_df, edgecolor=None, color=colors[0], s=10, zorder=1, ax=ax3, alpha=1, label=r'$P^C$')
sns.lineplot(x='x', y='y', data=P_C_df, color=colors[0], zorder=2, ax=ax3, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_C[0]], y=[opt_C[1]], edgecolor='white', color=colors[0], marker='o', s = 100, zorder=3, ax=ax3)

# Alternative
sns.scatterplot(x='x', y='y', data=P_A_df, edgecolor=None, color=colors[2], zorder=1, ax=ax3, s=10, alpha=1, label=r'$P^A$')
sns.lineplot(x='x', y='y', data=P_A_df, color=colors[2], zorder=2, ax=ax3, alpha=0.5, linewidth=0.5)
# plot optimal
sns.scatterplot(x=[opt_A[0]], y=[opt_A[1]], edgecolor='black', color=colors[2], marker='o', s = 100, zorder=3, ax=ax3)

# add legend
ax3.legend(loc='upper left', bbox_to_anchor=(1.05, 0.98), 
           borderaxespad=0., edgecolor='black', labelspacing=0).get_frame().set_boxstyle("square")

# add r labels
ax3.text(0.05, 0.95, r"$r_{z_{1}} = 0.8$", transform=ax3.transAxes, ha='left', va='top', fontsize=10)
ax3.text(0.05, 0.85, r"$r_{z_{2}} = 0.8$", transform=ax3.transAxes, ha='left', va='top', fontsize=10)

# tidying up
ax1.set_xlabel(r'$z_1$', fontsize=12)
ax1.set_ylabel(r'$z_2$', fontsize=12)
ax2.set_xlabel(r'$z_1$', fontsize=12)
ax3.set_xlabel(r'$z_1$', fontsize=12)

ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax2.set_xticklabels([])
ax3.set_xticklabels([])

ax1.set_title('Alternative bias')
ax2.set_title('Equal bias')
ax3.set_title('Conditional bias')
                                       
plt.tight_layout()
plt.show()