#!/usr/bin/env python3

import numpy as np
from scipy.stats import beta as beta_dist
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

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
alpha = 10 # --- trait bias towards conditional phenotype
beta = 10 # --- trait bias towards alternative phenotype
# get percentiles
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, n_traits)
r_init = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution
opt_C = r_init * z_0 # --- optimal phenotype for conditional 
opt_A = (1-r_init) * z_0 # --- optimal phenotype for alternative
# MUTATIONS
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 0.1 # --- standard deviation in mutation effect on z
sigma_r = 0.01 # --- standard deviation in mutation effect on r
# POPULATION/FITNESS
population_size = 100 # --- population size
sigma_s = 1 # --- strength of stabilizing selection
f_C = 0 # --- frequency/probability of conditional
f_A = 1 # --- frequency/probability of alternative
generations = 100

# SIMULATIONS
history = evolve_r(n_traits, z_0, r_init, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, sigma_r, f_C, f_A, generations)

# PLOTTING
fig, [ax1,ax2] = plt.subplots(1,2, figsize=(10,4))

# plot r distribution
sns.histplot(r_init, ax=ax1, color='tab:gray')
ax1.set_xlim(0,1)

# plot dynamics
sns.lineplot(history['mean_W_C'], ax=ax2, linestyle = '-', color='black', label = r'$W^C$')
sns.lineplot(history['mean_W_A'], ax=ax2, linestyle = '--', color='tab:gray', label = r'$W^A$')

ax2.set_ylabel('Fitness')
plt.tight_layout()
plt.show()