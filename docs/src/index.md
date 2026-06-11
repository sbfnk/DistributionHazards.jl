# DistributionHazards.jl

Hazard, cumulative hazard, survival, and density methods (with log-scale
variants) for any continuous univariate distribution in
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl), plus
[`expected_cumhazard`](@ref) via QuadGK.

Distributions.jl already provides `pdf`, `cdf`, `ccdf`, `logpdf`, `logccdf`,
and `quantile` for every univariate distribution. It does not expose hazard,
cumulative hazard, or survival as named functions, even though they are
trivially defined for any survival distribution. This package fills that gap,
plus an `iszero(t)` guard so ForwardDiff stays stable through `0^shape`
expressions at `t = 0` (Weibull, Gamma, log-logistic, …).

## At a glance

```julia
using DistributionHazards, Distributions

d = Weibull(1.7, 2.0)                # shape α = 1.7, scale θ = 2

hazard(d, 0.5)                       # h(t) = f(t) / S(t)
cumhazard(d, 0.5)                    # H(t) = -log S(t)
survival(d, 0.5)                     # S(t) = ccdf(d, t)
density(d, 0.5)                      # f(t) = pdf(d, t)

log_hazard(d, 0.5)                   # AD-stable
log_survival(d, 0.5)                 # = -cumhazard(d, t)
log_density(d, 0.5)

# Expected cumulative hazard of d1 under a second distribution d2.
expected_cumhazard(d, Exponential(1.0))   # E[H_d(X)], X ~ Exp(1), via QuadGK
```

[`expected_cumhazard(d1, d2)`](@ref) is `∫ H_{d1}(t) f_{d2}(t) dt`. In a
branching-process epidemic with `d1` a contact-interval distribution and
`d2` an infectious-period distribution, this is the basic reproduction
number.

## CensoredDistributions compatibility

Dispatch is on `Distributions.UnivariateDistribution`, so anything that
subtypes it works — including the censored distributions from
[CensoredDistributions.jl](https://github.com/epiaware/CensoredDistributions.jl):

```julia
using CensoredDistributions
d = primary_censored(Gamma(2.0, 1.5), Uniform(0.0, 1.0))
hazard(d, 1.5); cumhazard(d, 1.5); survival(d, 1.5)
```

The test suite includes an integration check that the methods dispatch on
`primary_censored(...)` and return finite, consistent values.

## What's in scope

Just the seven methods plus the helper, dispatched on
`Distributions.UnivariateDistribution`. No regression, no estimation, no
Kaplan–Meier / Nelson–Aalen — those live in the survival-analysis packages
([Survival.jl](https://github.com/JuliaStats/Survival.jl),
[LSurvival.jl](https://github.com/alexpkeil1/LSurvival.jl),
[SurvivalAnalysis.jl](https://github.com/vollmersj/SurvivalAnalysis.jl),
[HazReg.jl](https://github.com/FJRubio67/HazReg.jl)).

See the [API reference](@ref) for the full method list.
```@meta
CurrentModule = DistributionHazards
```
