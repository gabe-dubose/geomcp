import numpy as np

#########################
###### PARAMETERS #######
#########################

# N_TRAITS: number of traits to consider [float]
# Z_0: vector of initial trait (z) values. Must be = len(N_TRAITS) [numpy.array, float]
# R_0: vector of initial relative expression (r) values.  Must be = len(N_TRAITS) [numpy.array, float]
# OPT_C: vector of optimal values for elementwise z*r (conditional phenotype) [numpy.array, float]
# OPT_A: vector of optimal values for elementwise z*r (alternative phenotype) [numpy.array, float]
# SIGMA_M: standard deviation of normal distrubution used for z mutation effect generation [float]
# SIGMA_S: strength of stabilizing selection (how sharp fitness declines with distance from optimum) [float]
# SIGMA_R: standard devaition of normal distribtuion used for r mutation effect generation [float]
# F_C: relative frequency of conditional phenotype expression in population [float, <= 1]
# F_A: relative frequency of alternative phenotype expression in population [float, <= 1]
# POPULATION_SIZE: number of individuals in population [int]
# GENERATIONS: number of generations to conduct simulation [int]

####################
###### VALUE #######
####################

# DYNAMICS: dictionary containing mean poulation features over time [dictionary]
    # keys:
        # 'W_C': mean fitness of conditional phenotype [list, float]
        # 'W_A': mean fitness of alternative phenotype [list, float]
        # 'P_C': mean conditional phenotype values [list, float]
        # 'P_A': mean alternative phenotype values [list, float]
        # 'r': mean r values [array, float]
        # 'z': mean z values [array, float]


def evolve(N_TRAITS, Z_0, R_0, OPT_C, OPT_A, SIGMA_M, SIGMA_R, SIGMA_S, F_C, F_A, POPULATION_SIZE, GENERATIONS):

    # initialize vectors to track mean population characteristics
    mean_W_C = []
    mean_W_A = []
    mean_P_C = []
    mean_P_A = []
    mean_r = []
    mean_z = []

    # initialize population storage
    population = {}
    W_C_0 = []
    W_A_0 = []
    P_C_0 = []
    P_A_0 = []
    r_init = []
    z_init = []

    # simulate first generation
    for i in range(POPULATION_SIZE):
        individual_z = Z_0 + np.random.normal(0, SIGMA_M, N_TRAITS)
        individual_r = R_0.copy()
        P_C = individual_z * individual_r
        P_A = individual_z * (1-individual_r)
        dist_C = np.sum((P_C - OPT_C) ** 2)
        dist_A = np.sum((P_A - OPT_A) ** 2)
        W_C = np.exp(-SIGMA_S * dist_C)
        W_A = np.exp(-SIGMA_S * dist_C)
        W_T = W_C * F_C + W_A * F_A
        W_ = [W_T, W_C, W_A]
        population[i] = [individual_z, individual_r, W_]

        W_C_0.append(W_C)
        W_A_0.append(W_A)
        P_C_0.append(P_C)
        P_A_0.append(P_A)
        r_init.append(individual_r)
        z_init.append(individual_z)

    # record initial means
    mean_W_C.append(np.mean(W_C_0))
    mean_W_A.append(np.mean(W_A_0))
    mean_P_C.append(np.mean(P_C_0, axis=0))
    mean_P_A.append(np.mean(P_A_0, axis=0))
    mean_z.append(np.mean(z_init))
    mean_r.append(np.mean(r_init))

    # iterate through and generations and evolve population
    for generation in range(GENERATIONS):
        fitness_values = [population[i][2][0] for i in population]
        population_fitness = sum(fitness_values)
        relative_fitness = [individual_fitness / population_fitness for individual_fitness in fitness_values]
        parents = np.random.choice(list(population.keys()), size = POPULATION_SIZE, p = relative_fitness)
        next_generation = {}
        W_C_pop, W_A_pop, P_C_pop, P_A_pop, z_pop, r_pop = [], [], [], [], [], []

        for i, p in enumerate(parents):
            parent_z, parent_r, _ = population[p]
            delta_z = np.random.normal(0, SIGMA_M, N_TRAITS)
            delta_r = np.random.normal(0, SIGMA_R)
            offspring_z = parent_z + delta_z
            offspring_r = np.clip(parent_r + delta_r, 0, 1)

            offspring_P_C = offspring_z * offspring_r
            offspring_P_A = offspring_z * (1-offspring_r)
            dist_C = np.sum((offspring_P_C - OPT_C) ** 2)
            dist_A = np.sum((offspring_P_A - OPT_A) ** 2)
            W_C = np.exp(-SIGMA_S * dist_C)
            W_A = np.exp(-SIGMA_S * dist_A)
            W_T = W_C * F_C + W_A * F_A
            W_ = [W_T, W_C, W_A]

            next_generation[i] = [offspring_z, offspring_r, W_]

            W_C_pop.append(W_C)
            W_A_pop.append(W_A)
            P_C_pop.append(offspring_P_C)
            P_A_pop.append(offspring_P_A)
            r_pop.append(offspring_r)
            z_pop.append(offspring_z)

        mean_W_C.append(np.mean(W_C_pop))
        mean_W_A.append(np.mean(W_A_pop))
        mean_P_C.append(np.mean(P_C_pop, axis=0))
        mean_P_A.append(np.mean(P_A_pop, axis=0))
        mean_r.append(np.mean(r_pop))
        mean_z.append(np.mean(z_pop))

        population = next_generation

    DYNAMICS = {'W_C' : mean_W_C,
                'W_A' : mean_W_A,
                'P_C' : mean_P_C,
                'P_A' : mean_P_A,
                'r' : mean_r,
                'z' : mean_z}
    
    return(DYNAMICS)