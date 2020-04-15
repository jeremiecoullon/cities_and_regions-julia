## Cities and Regions code in Julia

Code for [Stochastic modelling or urban structures](https://royalsocietypublishing.org/doi/10.1098/rspa.2017.0700) ported to Julia. Original code (in Python) is [here](https://github.com/lellam/cities_and_regions/)


- `potential_2d.jl:` Illustration of 2d potential function to produce figure 2.
- `sample_prior_hmc.jl`: HMC code to generate data for figure 5. To use multiple threads for within-temperature moves, run `JULIA_NUM_THREADS=4 julia sample_prior_hmc.jl`
- `opt.jl`: Optimisation routine to generate data for figure 6.
- `rsquared_analysis.jl`: R-squared analysis for deterministic model as discussed around figure 4.
- `laplace_grid.jl`: Likelihood values for figure 4.
