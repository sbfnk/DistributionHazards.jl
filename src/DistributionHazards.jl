"""
DistributionHazards.jl — hazard, cumulative hazard, survival, density (and
log-scale variants) for any continuous univariate distribution in
[Distributions.jl](https://github.com/JuliaStats/Distributions.jl), plus
`expected_cumhazard(d1, d2)` via QuadGK.

Distributions.jl already provides `pdf`, `cdf`, `ccdf`, `logpdf`, `logccdf`,
and `quantile`. It does not expose hazard, cumulative hazard, or survival as
named functions, even though they are trivially defined for any survival
distribution. This package fills that gap, plus an `iszero(t)` guard so
ForwardDiff stays stable through `0^shape` expressions at `t = 0`.
"""
module DistributionHazards

import Distributions
using QuadGK: quadgk

export hazard, cumhazard, survival, density
export log_hazard, log_survival, log_density
export expected_cumhazard

const UD = Distributions.UnivariateDistribution

# Type-stable zero at t = 0 that promotes correctly through ForwardDiff.
# Computing the zero from `partype(d)` and `typeof(t)` avoids calling
# `logccdf(d, 0)`, whose 0^shape expression can NaN through duals.
_zero(d::UD, t::Real) = zero(promote_type(Distributions.partype(d), typeof(t)))

"""
    hazard(d, t)

Hazard `h(t) = f(t) / S(t)` of a continuous univariate distribution `d` at
analysis time `t`.

The generic method evaluates `exp(logpdf(d, t) - logccdf(d, t))` rather than
`pdf(d, t) / ccdf(d, t)` directly, so the value stays finite at extreme `t`
where both `pdf` and `ccdf` underflow to zero. Closed-form specialisations
exist for `Exponential`, `Weibull`, `Rayleigh`, `Pareto`, and `Frechet`.
"""
hazard(d::UD, t::Real) = exp(log_hazard(d, t))

"""
    log_hazard(d, t)

`log h(t) = log f(t) - log S(t)`. Evaluated via `logpdf` and `logccdf` so it
stays AD-stable on the log scale.
"""
log_hazard(d::UD, t::Real) =
    Distributions.logpdf(d, t) - Distributions.logccdf(d, t)

"""
    cumhazard(d, t)

Cumulative hazard `H(t) = -log S(t)`. Returns zero at `t = 0` to keep
ForwardDiff stable through distributions whose `logccdf` evaluates `0^shape`
at the boundary (Weibull, Gamma, log-logistic, …).
"""
function cumhazard(d::UD, t::Real)
    iszero(t) && return _zero(d, t)
    return -Distributions.logccdf(d, t)
end

"""
    survival(d, t)

Survival function `S(t) = 1 - F(t)`, equal to `Distributions.ccdf(d, t)`.
"""
survival(d::UD, t::Real) = Distributions.ccdf(d, t)

"""
    log_survival(d, t)

`log S(t) = -H(t)`. Evaluated as `-cumhazard(d, t)` so it inherits the
`t = 0` guard.
"""
log_survival(d::UD, t::Real) = -cumhazard(d, t)

"""
    density(d, t)

Probability density `f(t) = pdf(d, t)`.
"""
density(d::UD, t::Real) = Distributions.pdf(d, t)

"""
    log_density(d, t)

`log f(t) = logpdf(d, t)`.
"""
log_density(d::UD, t::Real) = Distributions.logpdf(d, t)

"""
    expected_cumhazard(d1, d2; atol=1e-10, rtol=1e-8) -> Float64

`E[H_{d1}(X)]` for `X ~ d2`, computed by adaptive Gauss–Kronrod quadrature
over the support of `d2`.

In branching-process epidemiology, with `d1` a contact-interval distribution
and `d2` an infectious-period distribution, this is the basic reproduction
number `R₀` in the no-depletion limit.
"""
function expected_cumhazard(d1::UD, d2::UD;
        atol::Real = 1e-10, rtol::Real = 1e-8)
    lo = max(0.0, Float64(Distributions.minimum(d2)))
    hi = Float64(Distributions.maximum(d2))
    val, _ = quadgk(t -> cumhazard(d1, t) * Distributions.pdf(d2, t),
                     lo, hi; atol = atol, rtol = rtol)
    return val
end

# ──────────────────────────────────────────────────────────────────────────
# Closed-form specialisations for common survival distributions.
#
# The generic methods above already evaluate `log_hazard` in log space and
# `cumhazard` as `-logccdf`, which Distributions.jl implements via the right
# algebra for each family. The specialisations below are useful because:
#
#   - They write the hazard / cumulative hazard / log hazard at the call
#     site, so the algebra is visible and the value is one shape away from
#     the textbook formula.
#   - They short-circuit the `iszero(t)` guard with the correct boundary
#     value for each family (typically zero, sometimes infinite).
#
# Per-family numerical-stability tests in the suite confirm the closed forms
# agree with the generic recipe at moderate `t` and stay finite where the
# `pdf / ccdf` recipe would underflow.
# ──────────────────────────────────────────────────────────────────────────

# Exponential(θ) — rate λ = 1/θ; h(t) = 1/θ, H(t) = t/θ.
hazard(d::Distributions.Exponential, t::Real) = inv(d.θ) * one(t)
function cumhazard(d::Distributions.Exponential, t::Real)
    iszero(t) && return _zero(d, t)
    return t / d.θ
end
log_hazard(d::Distributions.Exponential, t::Real) = -log(d.θ) * one(t)

# Weibull(α, θ) — h(t) = (α/θ)(t/θ)^(α-1), H(t) = (t/θ)^α.
function hazard(d::Distributions.Weibull, t::Real)
    iszero(t) && return _zero(d, t)
    α, θ = d.α, d.θ
    return (α / θ) * (t / θ)^(α - 1)
end
function cumhazard(d::Distributions.Weibull, t::Real)
    iszero(t) && return _zero(d, t)
    return (t / d.θ)^d.α
end
function log_hazard(d::Distributions.Weibull, t::Real)
    α, θ = d.α, d.θ
    return log(α / θ) + (α - 1) * log(t / θ)
end

# Rayleigh(σ) — equivalent to Weibull(2, σ√2); h(t) = t/σ², H(t) = t²/(2σ²).
hazard(d::Distributions.Rayleigh, t::Real) = t / d.σ^2
function cumhazard(d::Distributions.Rayleigh, t::Real)
    iszero(t) && return _zero(d, t)
    return (t / d.σ)^2 / 2
end
log_hazard(d::Distributions.Rayleigh, t::Real) = log(t) - 2 * log(d.σ)

# Pareto(α, θ) — support t ≥ θ; h(t) = α/t, H(t) = α·log(t/θ).
hazard(d::Distributions.Pareto, t::Real) = d.α / t
cumhazard(d::Distributions.Pareto, t::Real) = d.α * log(t / d.θ)
log_hazard(d::Distributions.Pareto, t::Real) = log(d.α) - log(t)

# Frechet(α, θ) — pdf(t) = (α/θ)(t/θ)^(-α-1) exp(-z), ccdf(t) = -expm1(-z)
# where z = (t/θ)^(-α). The `expm1` form keeps the ccdf accurate for small z
# (i.e. large t), where `1 - exp(-z)` would lose precision.
function hazard(d::Distributions.Frechet, t::Real)
    iszero(t) && return _zero(d, t)
    α, θ = d.α, d.θ
    z = (t / θ)^(-α)
    return (α / θ) * (t / θ)^(-α - 1) * exp(-z) / (-expm1(-z))
end
function cumhazard(d::Distributions.Frechet, t::Real)
    iszero(t) && return _zero(d, t)
    α, θ = d.α, d.θ
    z = (t / θ)^(-α)
    return -log(-expm1(-z))
end
function log_hazard(d::Distributions.Frechet, t::Real)
    α, θ = d.α, d.θ
    z = (t / θ)^(-α)
    return log(α / θ) + (-α - 1) * log(t / θ) - z - log(-expm1(-z))
end

end # module
