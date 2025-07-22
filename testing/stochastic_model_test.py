#!/usr/bin/env python3

# import stochastic model module
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from geomcp.stochastic_model import evolve

# import additional modules for testing
import numpy as np
from scipy.stats import beta as beta_dist
import matplotlib.pyplot as plt

#########################
###### PARAMETERS #######
#########################

# TRAIT DISTRIBUTION
N_TRAITS = 20
Z_0 = np.ones(N_TRAITS) # --- number of traits

# R DISTRIBUTION
alpha = 20
beta = 20
r_percentiles = np.linspace(1*10**-20, 1-1*10**-20, N_TRAITS)
R_0 = r_high = beta_dist.ppf(r_percentiles, alpha, beta) # --- r distribution

# OPTIMA
OPT_C = Z_0 * R_0 # --- optimal phenotype for conditional 
OPT_A = Z_0 * (1-R_0) # --- optimal phenotype for alternative

# EVOLUTION PARAMETERS
SIGMA_M = 0.1 # --- standard deviation in mutation effect on z
SIGMA_S = 1 # --- strength of stabilizing selection
SIGMA_R = 0 # --- standard deviation in mutation effect on r

# POPULATION PARAMETERS
F_C = 0 # --- frequency of conditional phenotype expression
F_A = 1 # --- frequency of alternative phenotype expression
POPULATION_SIZE = 1000 # --- population size
GENERATIONS = 50 # --- number of generations

#########################
###### SIMULATION #######
#########################

# run stimulation
dynamics = evolve(N_TRAITS, Z_0, R_0, OPT_C, OPT_A, SIGMA_M, SIGMA_R, SIGMA_S, F_C, F_A, POPULATION_SIZE, GENERATIONS)

############################
###### PLOT DYNAMICS #######
############################

plt.plot(dynamics['W_C'])
plt.xlabel('Generations')
plt.ylabel(r'$W^C$')
plt.show()