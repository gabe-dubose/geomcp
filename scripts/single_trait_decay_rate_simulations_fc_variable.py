#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist
from scipy.optimize import curve_fit

###########################
# --- EVOLVE FUNCTION --- #
###########################

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
                   

######################
# --- PARAMETERS --- #
######################

r_values = np.arange(0, 1.05, 0.01)
f_C_values = np.arange(0, 1.05, 0.01)
f_A = 1
n_traits = 1
z_0 = np.ones(n_traits) # --- initial trait values
# MUTATIONS
mu = 0 # --- mean effect of mutation (always 0)
sigma_m = 0.1 # --- standard deviation in affect
# POPULATION/FITNESS
population_size = 100 # --- population size
sigma_s = 1 # --- width of fitness peak 
generations = 50 # --- number of generations

#####################
# --- FUNCTIONS --- #
#####################

def exp_decay(t, W0, lamb):
    return W0 * np.exp(-lamb * t)

def fitness_decay(n_runs, r, f_C):
    
    # define internal paramaters
    opt_C = r * z_0
    opt_A = (1-r) * z_0
    
    # iteratively run model
    for run in range(n_runs):
        if run == 0:
            history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            history_df = pd.DataFrame(history['mean_W_C'])
        else:
            history = evolve(n_traits, z_0, r, opt_C, opt_A, mu, sigma_m, population_size, sigma_s, f_C, f_A, generations)
            history_df[f"run_{run}"] = history['mean_W_C']

    # get mean function
    mean_dynamics = history_df.mean(axis = 1)
    generations_array = np.arange(len(mean_dynamics))

    # fit exponential decay model to mean fitness data
    popt, pcov = curve_fit(exp_decay, generations_array, mean_dynamics, p0=(1, 0.01))
    W0_fit, lambda_fit = popt

    return {
        "mean_dynamics": mean_dynamics,
        "W0_fit": W0_fit,
        "lambda_fit": lambda_fit,
        "history_df": history_df
    }

def decay_rate_space(r_values, f_C_values, n_runs):
    lambda_grid = pd.DataFrame(index=np.round(r_values, 2), columns=np.round(f_C_values, 2))

    for r in r_values:
        for f_C in f_C_values:
            try:
                output = fitness_decay(n_runs, r, f_C)
                lambda_val = output["lambda_fit"]
            except Exception as e:
                lambda_val = np.nan  # fallback if fit fails
            lambda_grid.loc[round(r, 2), round(f_C, 2)] = lambda_val
            
        print(f"Completed anlalysis of r = {r}")
        
    return lambda_grid.astype(float)

n_runs = 10
decay_rates = decay_rate_space(r_values, f_C_values, n_runs)
decay_rates.to_csv('decay_rates_data.csv')
