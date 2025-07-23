#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta as beta_dist
import seaborn as sns
import matplotlib.pyplot as plt
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

# PARAMETERS
# PHENOTYPE
n_traits = 20 # --- number of traits
z_0 = np.ones(n_traits) # --- initial trait values
alpha = 5 # --- trait bias towards conditional phenotype
beta = 5 # --- trait bias towards alternative phenotype
# get percentiles
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, n_traits)
r = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution
opt_C = r * z_0 # --- optimal phenotype for conditional 
opt_A = (1-r) * z_0 # --- optimal phenotype for alternative
# MUTATIONS
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 0.1 # --- standard deviation in affect
# POPULATION/FITNESS
population_size = 100 # --- population size
sigma_s = 1 # --- width of fitness peak 
f_C = 0 # --- frequency/probability of conditional
f_A = 1 # --- frequency/probability of alternative
generations = 100

history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)

# PLOTTING

fig, [ax1,ax2] = plt.subplots(1,2, figsize=(10,4))

# plot r distribution
sns.histplot(r, ax=ax1, color='tab:gray')
ax1.set_xlim(0,1)

# plot dynamics
sns.lineplot(history['mean_W_C'], ax=ax2, linestyle = '-', color='black', label = r'$W^C$')
sns.lineplot(history['mean_W_A'], ax=ax2, linestyle = '--', color='tab:gray', label = r'$W^A$')

ax2.set_ylabel('Fitness')
plt.tight_layout()
plt.show()
