#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta as beta_dist
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.cm as cm
import seaborn as sns
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
# PHENOTYPE
n_traits = 20 # --- number of traits
z_0 = np.ones(n_traits) # --- initial trait values
# MUTATIONS
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 0.1 # --- standard deviation in affect
# POPULATION/FITNESS
population_size = 1000 # --- population size
sigma_s = 1 # --- width of fitness peak 
generations = 50 # --- number of generations
f_C = 0 # --- frequency/probability of conditional
f_A = 1 # --- frequency/probability of alternative

# Variable r distributions
# r distributions to consider
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

def iterate_simulations(n_runs):
    for i in range(n_runs):

        if i == 0:
            # run and initialize dataframes
            r_no_history = evolve(n_traits, z_0, r_no, opt_C_no, opt_A_no, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            r_no_history_df = pd.DataFrame(r_no_history['mean_W_C'])
            # run and initialize dataframes
            r_low_history = evolve(n_traits, z_0, r_low, opt_C_low, opt_A_low, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            r_low_history_df = pd.DataFrame(r_low_history['mean_W_C'])
            # run and initialize dataframes
            r_high_history = evolve(n_traits, z_0, r_high, opt_C_high, opt_A_high, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            r_high_history_df = pd.DataFrame(r_high_history['mean_W_C'])            
        else:
            # run
            r_no_history = evolve(n_traits, z_0, r_no, opt_C_no, opt_A_no, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            r_low_history = evolve(n_traits, z_0, r_low, opt_C_low, opt_A_low, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            r_high_history = evolve(n_traits, z_0, r_high, opt_C_high, opt_A_high, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            
            r_no_history_df[f"run_{i}"] = r_no_history['mean_W_C']
            r_low_history_df[f"run_{i}"] = r_low_history['mean_W_C']
            r_high_history_df[f"run_{i}"] = r_high_history['mean_W_C']

    return {'r_no' : r_no_history_df, 
           'r_low' : r_low_history_df, 
           'r_high' : r_high_history_df}

# run simulations
results = iterate_simulations(25)

# define colors
n_col = 3
viridis = cm.get_cmap('viridis', n_col)
colors = [viridis(i) for i in range(n_col)]

# set up plot
fig = plt.figure(figsize=(6, 4))
gs = gridspec.GridSpec(3, 2, width_ratios=[1, 2.5])

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[:, 1])

# plot histograms of r
sns.histplot(r_no, ax=ax1, binwidth=0.05, stat='proportion', color = colors[2], edgecolor='black')
sns.histplot(r_low, ax=ax2, binwidth=0.05, stat='proportion', color = colors[1], edgecolor='black')
sns.histplot(r_high, ax=ax3, binwidth=0.05, stat='proportion', color = colors[0], edgecolor='black')

# tidy-up
ax1.set_xlim(0,1)
ax2.set_xlim(0,1)
ax3.set_xlim(0,1)
ax1.set_ylim(0,0.5)
ax2.set_ylim(0,0.5)
ax3.set_ylim(0,0.5)
ax3.set_xlabel(r'$r$')

# plot dynamics
# no pleio
for column in results['r_no'].columns:
    sns.lineplot(results['r_no'][column].values, ax=ax4, linestyle = '-', color=colors[2], alpha=0.25)
# plot mean
sns.lineplot(results['r_no'].mean(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[2], linewidth=2, label = 'No pleiotropy')

# low pleio
for column in results['r_low'].columns:
    sns.lineplot(results['r_low'][column].values, ax=ax4, linestyle = '-', color=colors[1], alpha=0.25)
# plot mean
sns.lineplot(results['r_low'].mean(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[1], linewidth=2, label = 'Low pleiotropy')

# high high
for column in results['r_high'].columns:
    sns.lineplot(results['r_high'][column].values, ax=ax4, linestyle = '-', color=colors[0], alpha=0.25)
# plot mean
sns.lineplot(results['r_high'].mean(axis=1).to_list(), ax=ax4, linestyle = '-', color=colors[0], linewidth=2, label = 'High pleiotropy')

ax4.set_ylabel(r'Mean $W^C$')
ax4.set_xlabel('Generation')
ax4.set_ylim(0,1)

# add subpanel labels
#ax1.text(-0.92, 1.03, 'A', transform=ax.transAxes,
#            fontsize=14, va='top', ha='right')
#ax2.text(-0.92, 0.655, 'B', transform=ax.transAxes,
#            fontsize=14, va='top', ha='right')
#ax3.text(-0.92, 0.275, 'C', transform=ax.transAxes,
#            fontsize=14, va='top', ha='right')
#ax4.text(-0.1, 1.03, 'D', transform=ax.transAxes,
#            fontsize=14, va='top', ha='right')

plt.tight_layout()
plt.show()