[![DOI](https://zenodo.org/badge/996437834.svg)](https://doi.org/10.5281/zenodo.15608125)
# GEOMCP: GEOmetric Model of Conditional Phenotypes
This repository contains the scripts and simulated data used in the associated manuscript (<i>Geometric insights into the role of pleiotropy in shaping the evolutionary dynamics of conditional phenotypes</i>). Model descriptions, as well as descriptions of simulation and numerical analyses can be foudn in the associated manuscript. As described below, allo scripts are located in the <b><i>scripts</i></b> directory, and simulated data is located in the  <b><i>scripts</i></b> directory. 

## scripts

```basic_model.py```: contains code to run and plot most basic model formulation (does not include functionality for an evolving pleiotropic architecture).

```phenotype_space_model_illustration.py```: contains the model from basic_model.py, but plots the population's mean phenotype (limited to 2 traits) throughout evolutionary simulation. This script was used to create Figure 1 in the associated manuscript. 

```non_evolving_pleio_conditional_maintenance_simulations.py```: contains the model from basic_model.py, but includes functionality for simulating and plotting replicate evolutionary trajectories. It also accomodates evaluation of three different pleiotorpic architectures. This script was used to create Figure 2 in the associated manuscript. 

```single_trait_decay_rate_simulations_fc_variable.py```: runs model from basic_model.py for single traits across the $r$ and $f_C$ parameter space. Script runs each parameter combination n_runs = 10 times, and then fits an exponential decay function to the mean resulting dynamics (change in conditional fitness). The estimated decay rate is then saved. This script was used to generate the data presented in Figure 3 in the associated manuscript. 

```analyze_single_trait_decay_rate_simulations_fc_variable.py```: analyzes simulated data generated by single_trait_decay_rate_simulations_fc_variable.py, which includes Gaussian smoothing of the decay rates across the parameter space, estimation of decay rate as a function of expression frequency ($f_C$), and evaluating the derivative of said function. This script was used to generate Figure 3 in the associate manuscript.

```single_trait_decay_rate_simulations_g_variable.py```: similar to <i>single_trait_decay_rate_simulations_fc_variable.py</i>, script runs model from basic_model.py for single traits across the $r$ and $g$ parameter space. Script runs each parameter combination n_runs = 10 times, and then fits an exponential decay function to the mean resulting dynamics (change in conditional fitness). The estimated decay rate is then saved. This script was used to generate the data presented in Figure 4 in the associated manuscript. 

```analyze_single_trait_decay_rate_simulations_g_variable.py```: analyzes simulated data generated by single_trait_decay_rate_simulations_fc_variable.py, which includes Gaussian smoothing of the decay rates across the parameter space, estimation of decay rate as a function of generations between expression ($g$), and evaluating the derivative of said function. This script was used to generate Figure 4 in the associate manuscript.

```evolving_r_model.py```: contains code to run and plot most basic model formulation that does include functionality to allow for an evolving pleiotropic architecture.

```evolving_pleio_conditional_maintenance_simulations.py```: contains the model from evolving_r_model.py, but includes functionality for simulating and plotting replicate evolutionary trajectories. It also accomodates side-by-side evaluation of different standard devaition in $r$ mutationl effect size ($\sigma_r$), as well as assessment of variation in evolutionary trajectories associated with the simulations. Furthermore, this script displays the evolved pleiotorpic architectures (distributions of $r$) with reference to thier ancestral state, as well as the fitness landscape in terms of r and z associated with several alternative phenotype optima. This script was used to create Figure 5 and Figure 6 in the associated manuscript. 

## data

```decay_rates_data.csv```: a matrix in comma-separated format that contains estimated decay rates from simulations across the $r$ and $f_C$ parameter space. Columns represent expression frequency ($f_C$), and rows represent relative expression ($r$). This simulated data was generated by the <i>single_trait_decay_rate_simulations_fc_variable.py</i> scripts. 

```decay_rates_data_temporal.csv```: a matrix in comma-separated format that contains estimated decay rates from simulations across the $r$ and $g$ parameter space. Columns represent the number of generations between expression ($g$), and rows represent relative expression ($r$). This simulated data was generated by the <i>single_trait_decay_rate_simulations_g_variable.py</i> scripts. 
