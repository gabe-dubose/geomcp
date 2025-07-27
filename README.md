[![DOI](https://zenodo.org/badge/996437834.svg)](https://doi.org/10.5281/zenodo.15608125)
# GEOMCP: GEOmetric Model of Conditional Phenotypes
GEOMCP is an extension of Fisher's geometric model that allows for modeling of conditional phenotypes and their pleiotropic architectures. By including specification of expression frequencies of conditional and alternative phenotypes, this model provides insight into how pleiotorpic architecture and expression frequencies (both spatial and temporal) interact to shape the evolutionary dynamics of conditional phenotypes. To faciliate model exploration, GEOMCP is available via a Python package and a command line interface. 

## Installation
The <i>geomcp</i> Python module can be install using
```
pip3 install git+https://github.com/gabe-dubose/geomcp.git
```
The GEOMCP Python package relies on ```NumPy``` for various vector and mathematical operations, and ```Matplotlib``` for plotting. The latest versions of these packages that are compatible with the current Python environment are installed automatically.

## stochastic_model
GEOMCP at its core is a stochastic model, meaning evolutioanry dynamics are probabilistically simulated. Therefore, this module contains the functionality for constructing and simulating stochastic GEOMCP models. 

### stochastic_model.evolve

```
Parameters:
  N_TRAITS          Number of traits that compose the conditional phenotype: int
  Z_0               Initial trait values: ndarray of length N_TRAITS
  R_0               Initial relative expression values: ndarray of length N_TRAITS
  OPT_C             Optimal trait value (Z * R) for conditional phenotype: ndarray of length N_TRAITS
  OPT_A             Optimal trait value (Z * (1-R)) for alternative phenotype: ndarray of length N_TRAITS
  SIGMA_M           Standard deviation in mutational effect on Z: float or int
  SIGMA_R           Standard deviation in mutational effect on R: float or int
  SIGMA_S           Strength of stabilizing selection: float or int
  F_C               Expression frequency of conditional phenotyype: 1 or float < 1
  F_A               Expression frequency of alternative phenotyype: 1 or float < 1
  POPULATION_SIZE   Number of individuals in the simulated population:  int < 0
  GENERATIONS       Number of generations to run simulation: int < 0

Returns:
  dynamics          Dictionary with the keys:
                      'W_C'      mean fitness of conditional phenotype: list of floats
                      'W_A'      mean fitness of alternative phenotype: list of floats
                      'P_C'      mean conditional phenotype values: list of floats
                      'P_A':     mean alternative phenotype values: list of floats
                      'r':       mean r value: list of floats
                      'z':       mean z value: list of floats
```

Example:
```
from geomcp import stochastic_model
dynamics = stochastic_model.evolve(N_TRAITS, Z_0, R_0, OPT_C, OPT_A, SIGMA_M, SIGMA_R, SIGMA_S, F_C, F_A, POPULATION_SIZE, GENERATIONS)
```
