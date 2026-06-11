# DistributionHazards.jl

| **Documentation** | **Build Status** | **Code Quality** | **License** |
|:-----------------:|:----------------:|:----------------:|:-----------:|
| [![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://sbfnk.github.io/DistributionHazards.jl/dev/) | [![Test](https://github.com/sbfnk/DistributionHazards.jl/actions/workflows/test.yaml/badge.svg?branch=main)](https://github.com/sbfnk/DistributionHazards.jl/actions/workflows/test.yaml) [![codecov](https://codecov.io/gh/sbfnk/DistributionHazards.jl/graph/badge.svg)](https://codecov.io/gh/sbfnk/DistributionHazards.jl) | [![Aqua QA](https://raw.githubusercontent.com/JuliaTesting/Aqua.jl/master/badge.svg)](https://github.com/JuliaTesting/Aqua.jl) | [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) |

*Hazards, survival and density for Distributions.jl.*


## Why DistributionHazards.jl?

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) gives every distribution `pdf`, `cdf`, `ccdf`, `logpdf`, `logccdf` and `quantile`. It does not give them `hazard`, `cumhazard`, or `survival`. This package adds them.
- The methods dispatch on `Distributions.UnivariateDistribution`, so they work on plain Distributions.jl types and on anything that subtypes them. The censored distributions in [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl) are the obvious example.
- Closed-form `hazard`, `cumhazard`, and `log_hazard` ship for `Exponential`, `Weibull`, `Rayleigh`, `Pareto`, and `Frechet`. These stay finite at extreme `t` where the textbook `pdf(d, t) / ccdf(d, t)` recipe underflows to `NaN` (e.g. `hazard(Weibull(2, 1), 100) == 200`, where `pdf` and `ccdf` are both zero in Float64).
- The generic fallback for `hazard` evaluates `exp(logpdf(d, t) - logccdf(d, t))` instead of `pdf / ccdf`, so even distributions without a specialisation get the log-space stability.
- An `iszero(t)` guard on `cumhazard` returns a properly-promoting zero, which keeps ForwardDiff stable through distributions whose `logccdf` evaluates `0^shape` at `t = 0` (Weibull, Gamma, log-logistic).
- `expected_cumhazard(d1, d2)` integrates `H_{d1}(t) f_{d2}(t)` by adaptive Gauss-Kronrod quadrature. In a branching-process epidemic, that's `R₀`.


## Getting started

```julia
using DistributionHazards, Distributions

d = Weibull(1.7, 2.0)               # shape α = 1.7, scale θ = 2

hazard(d, 0.5)                      # h(t) = f(t) / S(t)
cumhazard(d, 0.5)                   # H(t) = -log S(t)
survival(d, 0.5)                    # S(t) = ccdf(d, t)
density(d, 0.5)                     # f(t) = pdf(d, t)

log_hazard(d, 0.5)                  # = logpdf(d, t) - logccdf(d, t)
log_survival(d, 0.5)                # = -cumhazard(d, t)
log_density(d, 0.5)                 # = logpdf(d, t)
```

The expected cumulative hazard of `d1` under a second distribution `d2`:

```julia
expected_cumhazard(d, Exponential(1.0))   # E[H_d(X)], X ~ Exp(1), via QuadGK
```

ForwardDiff is stable through the parameters of `d` and at `t = 0`:

```julia
using ForwardDiff
ForwardDiff.gradient(p -> cumhazard(Weibull(p[1], p[2]), 0.0), [1.5, 2.0])
# 2-element Vector{Float64}: 0.0  0.0
```

With [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl):

```julia
using CensoredDistributions
d = primary_censored(Gamma(2.0, 1.5), Uniform(0.0, 1.0))
hazard(d, 1.5); cumhazard(d, 1.5); survival(d, 1.5)
```


## Relationship to Distributions.jl

Most of the quantities here already exist in Distributions.jl under different names. The rest are one-liners on top of `pdf` and `ccdf`:

| Survival name | Distributions.jl equivalent |
|---|---|
| `survival(d, t)` | `ccdf(d, t)` |
| `density(d, t)` | `pdf(d, t)` |
| `log_density(d, t)` | `logpdf(d, t)` |
| `cumhazard(d, t)` | `-logccdf(d, t)` (with `t = 0` guard) |
| `log_survival(d, t)` | `logccdf(d, t)` (with `t = 0` guard) |
| `hazard(d, t)` | `exp(logpdf(d, t) - logccdf(d, t))` (closed form for `Exponential`, `Weibull`, `Rayleigh`, `Pareto`, `Frechet`) |
| `log_hazard(d, t)` | `logpdf(d, t) - logccdf(d, t)` (closed form for the families above) |

If you have a `Distributions.UnivariateDistribution`, `using DistributionHazards` gives you the survival vocabulary for it.


## What packages work well with DistributionHazards.jl?

- [Distributions.jl](https://github.com/JuliaStats/Distributions.jl) is what everything dispatches on.
- [CensoredDistributions.jl](https://github.com/EpiAware/CensoredDistributions.jl) for primary-event and interval censoring. Its types subtype `UnivariateDistribution`, so the hazard methods work on them with no further code. There's an integration test in the suite.
- For Cox / Kaplan-Meier / Nelson-Aalen / parametric hazard regression, use [Survival.jl](https://github.com/JuliaStats/Survival.jl), [LSurvival.jl](https://github.com/alexpkeil1/LSurvival.jl), [SurvivalAnalysis.jl](https://github.com/vollmersj/SurvivalAnalysis.jl), or [HazReg.jl](https://github.com/FJRubio67/HazReg.jl). This package doesn't.


## Where to learn more

- [API reference](https://sbfnk.github.io/DistributionHazards.jl/dev/api/) for every exported function and its formula.
- [Issues](https://github.com/sbfnk/DistributionHazards.jl/issues) for bugs and feature requests.
- [Source](https://github.com/sbfnk/DistributionHazards.jl/).
