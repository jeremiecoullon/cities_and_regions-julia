## Cities and Regions code in Julia

Code for [Stochastic modelling or urban structures](https://royalsocietypublishing.org/doi/10.1098/rspa.2017.0700) ported to Julia. Original code (in Python) is [here](https://github.com/lellam/cities_and_regions/)

- `hmc.jl`: HMC code to generate data for figure 5. To use multiple threads for within-temperature moves, run `JULIA_NUM_THREADS=4 julia hmc.jl`
- `laplace_grid.jl`: Likelihood values for figure 4.
<!-- - `mcmc_high_noise.jl:` MCMC scheme for gamma=10000 (low-noise regime) to generate data for figures 9-10. -->
<!-- - `mcmc_low_noise.jl:` MCMC scheme for gamma=100 (high-noise regime) to generate data for figures 7-8. -->
- `opt.jl`: Optimisation routine to generate data for figure 6.
- `potential_2d.jl:` Illustration of 2d potential function to produce figure 2.
- `rsquared_analysis.jl`: R-squared analysis for deterministic model as discussed around figure 4.
- `data/london_n/`: Directory containing datasets for the case study.  Residential data is residential.csv and retail data is small_london.txt.  Remaining files are pre-processed versions for simulations.
