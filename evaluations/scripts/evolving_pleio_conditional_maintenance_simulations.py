#!/usr/bin/env python3

from geomcp import stochastic_model
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_dist

#############################
###### SET PARAMETERS #######
#############################

# TRAIT DISTRIBUTION
N_TRAITS = 20
Z_0 = np.ones(N_TRAITS) # --- number of traits

# EVOLUTION PARAMETERS
SIGMA_M = 0.1 # --- standard deviation in mutation effect on z
SIGMA_S = 1 # --- strength of stabilizing selection

# POPULATION PARAMETERS
F_C = 0 # --- frequency of conditional phenotype expression
F_A = 1 # --- frequency of alternative phenotype expression
POPULATION_SIZE = 1000 # --- population size
GENERATIONS = 50 # --- number of generations

# R DISTRIBUTION
alpha = 20
beta = 20
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, N_TRAITS)
R_0 = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution

# OPTIMA
OPT_C = Z_0 * R_0 # --- optimal phenotype for conditional 
OPT_A = Z_0 * (1-R_0) # --- optimal phenotype for alternative

# Non-evolving r
SIGMA_R = 0 # --- standard deviation in mutation effect on r
# simulation runs
# initialize results vector
results = []
iterations = 25
for i in range(iterations):
    dynamics = stochastic_model.evolve(N_TRAITS, Z_0, R_0, OPT_C, OPT_A, SIGMA_M, SIGMA_R, SIGMA_S, F_C, F_A, POPULATION_SIZE, GENERATIONS)
    mean_WC = dynamics['W_C']
    # add simulation run to results
    results.append(mean_WC)

# convert results to pandas dataframe
results_df = pd.DataFrame(results)
# save to file
results_df.to_csv('../data/non_evolving_pleio_conditional_decay.csv', index=False)

# Evolving r
SIGMA_R = 0.01 # --- standard deviation in mutation effect on r
# simulation runs
# initialize results vector
results = []
iterations = 25
for i in range(iterations):
    dynamics = stochastic_model.evolve(N_TRAITS, Z_0, R_0, OPT_C, OPT_A, SIGMA_M, SIGMA_R, SIGMA_S, F_C, F_A, POPULATION_SIZE, GENERATIONS)
    mean_WC = dynamics['W_C']
    # add simulation run to results
    results.append(mean_WC)

# convert results to pandas dataframe
results_df = pd.DataFrame(results)
# save to file
results_df.to_csv('../data/evolving_pleio_conditional_decay.csv', index=False)