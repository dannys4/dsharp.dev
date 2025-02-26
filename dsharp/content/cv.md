+++
title="Curriculum Vitae"
+++
# Short research statement
I am a doctoral researcher in the [MIT Uncertainty Quantification Group](https://uqgroup.mit.edu/) since 2021, and have tackled problems throughout uncertainty quantification (UQ). At the heart of my interest is the question of "how do we make UQ more efficient and effective for the average practitioner?" I tackle this through the lens of function approximation and "transport maps", which act as a deterministic change-of-variable translating UQ on a simple reference distribution (e.g., Gaussian or uniform) to a complicated target distribution, via a deterministic function. A few recent questions I have include:
- When learning these transport maps from unnormalized densities, how do we minimize forward evaluations of the target density? Can we make this process more parallel and efficient than, e.g., MCMC?
- Given a dataset of millions of points and a model I'd like to average over such a dataset, can I somehow "condense" this information and approximate the average with only a few hundred model evaluations?
- Can we extend "optimal sampling methods" for traditional UQ surrogates (e.g., polynomial chaos expansions) to the realm of more modern surrogate methods?

Broadly embracing these themes, I emphasize that my focus on "efficient and effective UQ" is not limited to efficient sampling, but also includes developing tools that domain experts can _actually use_ (via software, documentation, and tutorials) and ensuring that these tools are _robust_ and _reliable_ (both theoretically and practically).

# Education
## Massachusetts Institute of Technology, 2021-Present
### Computational Science and Engineering, S.M. 2023
> Master's thesis: **[Parameterizing transport maps for ensemble data assimilation](https://dspace.mit.edu/handle/1721.1/152488)**

## Virginia Tech, 2017-2021
> Computational Modeling and Data Analytics (CMDA), B.S. 2021
>
> Applied Computational Mathematics, B.S. 2021

# Publications
### Preprints
- Gradient-free multi-fidelity Bayesian inference via importance-weighted transport maps (To appear!)
- [Weighted quantization using MMD: From mean field to mean shift via gradient flows](https://arxiv.org/abs/2502.10600)

### Proceedings
- [A More Portable HeFFTe: Implementing a Fallback Algorithm for Scalable Fourier Transforms](https://ieeexplore.ieee.org/document/9622811) (IEEE HPEC, 2021)

### Journal Articles
> Coming soon!

### Non-primary authorship
- An introduction to triangular transport (Forthcoming!)
- [MParT: Monotone Parameterization Toolkit](https://joss.theoj.org/papers/10.21105/joss.04843.pdf) (JOSS, 2022)
- [Stably accelerating stiff quantitative systems pharmacology models: Continuous-time echo state networks as implicit machine learning](https://www.biorxiv.org/content/10.1101/2021.10.10.463808v1.full.pdf) (IFAC, 2022)

# Selected Honors
- CMDA Outstanding Senior Award, 2021
- Hamlett Scholar, 2018-2021
- Virginia Tech Math Department [Layman prize](https://math.vt.edu/math-news/news-2021/news-laymanwinners.html) winner (_Fourier transforms on the Modern Computer_), 2021
- Mathematical Contest in Modeling, Honorable Mention (top 8% of 3500+ teams internationally), 2020

# Work(-related) Experience
## Computational Science Intern, Sandia National Laboratories, 2023 - Present

### Gradient-free multi-fidelity Bayesian inference via importance-weighted transport maps (2024-present)
> Coordinated the conjunction of several methods within the realm of nonlinear transport to learn a Bayesian posterior distribution via sequence of transport maps, learned using annealed importance sampling. Since the maps are parametric and we do this approximation in discrete steps (as opposed to a continuous annealed flow), we are able to take advantage of multiple likelihood model. Further, the approximations made here are amenable to evaluation re-use via multiple importance sampling. This work is forthcoming.

### Simulation of Magnetohydrodynamics (2023)
> Worked with adjoint-enabled PDE simulation tool [MrHyDE](https://github.com/sandialabs/MrHyDE) to implement a simplified model of magnetohydrodynamics. Used this for performing gradient-enabled Bayesian inference on the model via Hamiltonian Monte Carlo.

## [Aerospace Computational Science and Engineering Lab](https://acdl-web.mit.edu) Sysadmin, 2022-2023
> Maintained and improved cutting-edge research compute hardware, including managing an 18-node slurm cluster as well has more than 40 personal workstations. Built and repaired computer hardware/firmware, provided technical support for lab members, and managed acquisitions.

# Talks and Presentations
- _Gradient-free multi-fidelity Bayesian inference via importance-weighted transport maps_, Sandia Uncertainty Quantification working group, 2024
- [_MParT: Monotone Parameterization Toolkit_](/posts/siam-uq24/), [SIAM UQ](https://meetings.siam.org/sess/dsp_programsess.cfm?SESSIONCODE=70292), 2024
- [_Creating transport maps with MParT.jl_](https://www.youtube.com/watch?v=eA24L_-a15I), [JuliaCon](https://juliacon.org/2023/), 2023
- _Parameterizing transport maps for sampling and Bayesian computation_, [MIT ACSEL Seminar series](https://acdl-web.mit.edu/seminars), 2023
- [_A More Portable HeFFTe: Implementing a Fallback Algorithm for Scalable Fourier Transforms_](https://ieeexplore.ieee.org/document/9622811), [IEEE HPEC](https://www.ieee-hpec.org/), 2021
- _Fourier transforms on the modern computer_, [Virginia Tech Layman Prize talks](https://math.vt.edu/math-news/news-2021/news-laymanwinners.html), 2021

# Teaching
- Stochastic Modeling and Inference (16.940), Graduate Teaching Assistant, MIT, Fall 2023
    - Topics include Monte Carlo methods (importance sampling, control variates), traditional surrogate construction (polynomial chaos expansions, pseudospectral methods), and Bayesian inference (Markov chain Monte Carlo, ensemble Kalman filter).
- Data Structures and Algorithms (CS3114), Teaching Assistant, Virginia Tech, Spring 2021
    - Topics include algorithm analysis, data structures (trees, graphs, hash tables), and algorithm design paradigms (divide-and-conquer, dynamic programming).

# Selected software
### Research Software
- [MParT](https://github.com/MeasureTransport/MParT): Perform significant amount of maintenance and development on this package, which provides monotone parameterizations for conditional modeling and inference.
- [FFTA.jl](https://github.com/dannys4/FFTA.jl): A pure-Julia implementation of the Fast Fourier Transform, compatible with GPU, automatic differentiation, and alternative datatypes.
- [MrHyDE](https://github.com/sandialabs/MrHyDE)*: Created a version of a C++ PDE solver that works as a cmake-compatible library, with convenient bindings for Python and Julia to evaluate parameterized PDEs.
- SmallMCMC.jl, EnsembleFiltering.jl*: Julia package for simple MCMC (including adaptive Metropolis) and ensemble filtering (including EnKF and ETKF) methods, respectively. Pure Julia enables use with GPU and automatic differentiation.
- Multiindexing.jl, MultivariateExpansions.jl, SparseQuadrature.jl*: Julia packages for efficient manipulation of multi-index sets, multivariate polynomials, and sparse quadrature rules for "classical" uncertainty quantification methods.
*_These codebases (or my contributions) may not be publicly available. Feel free to contact me for details!_

### Software for courses
- MIT 16.930, _Advanced Topics in PDEs_: Implementation of a hybridized discontinuous Galerkin method for the time-dependent convection-diffusion equation, using Python/JAX for automatic differentiation and GPU acceleration.
- MIT 6.1060, _Software Performance Engineering_: Developed highly performant parallelized OpenCILK/C codes for matrix-matrix multiplication, graphics rasterization, and a chess engine.
- Virginia Tech CMDA4984, _Advanced computational methods for CMDA_: Developed a MPI+CUDA/C implementation of a distributed and GPU-parallelized two-dimensional finite difference solver for the heat equation.

# Check out my [blog posts](/posts) for interesting projects relating to my work!