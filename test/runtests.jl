using DistributionHazards
using Aqua
using CensoredDistributions
using Distributions
using ForwardDiff
using Test

@testset "DistributionHazards.jl" begin
    @testset "Aqua" begin
        Aqua.test_all(DistributionHazards)
    end

    distributions = (
        Exponential(2.0),
        Weibull(1.7, 0.5),
        Weibull(0.8, 2.0),
        Gamma(2.5, 1.0),
        Gamma(0.6, 1.5),
        LogNormal(0.0, 0.5),
        Frechet(2.0, 1.0),
    )

    @testset "definitional consistency: $(typeof(d).name.name)$(Distributions.params(d))" for d in distributions
        for t in (0.1, 0.5, 1.0, 2.0, 5.0)
            @test cumhazard(d, t) ≈ -log_survival(d, t)
            @test density(d, t) ≈ hazard(d, t) * survival(d, t) atol = 1e-15 rtol = 1e-12
            @test log_density(d, t) ≈ log_hazard(d, t) + log_survival(d, t) atol = 1e-12
            @test exp(log_hazard(d, t)) ≈ hazard(d, t) rtol = 1e-10
            @test survival(d, t) ≈ ccdf(d, t)
            @test density(d, t) ≈ pdf(d, t)
        end
    end

    @testset "t = 0 returns the survival-distribution boundary" begin
        for d in distributions
            @test cumhazard(d, 0.0) == 0
            @test survival(d, 0.0) == 1
            @test log_survival(d, 0.0) == 0
        end
    end

    @testset "Exponential closed form" begin
        # X ~ Exponential(θ): H(t) = t/θ, h(t) = 1/θ
        for θ in (0.5, 1.0, 2.0), t in (0.5, 1.5, 3.0)
            d = Exponential(θ)
            @test cumhazard(d, t) ≈ t / θ
            @test hazard(d, t) ≈ 1 / θ
        end
    end

    @testset "Weibull closed form" begin
        # Distributions.Weibull(α, θ): S(t) = exp(-(t/θ)^α), so H(t) = (t/θ)^α.
        for α in (0.7, 1.0, 2.5), θ in (0.5, 1.0), t in (0.3, 1.0, 2.5)
            d = Weibull(α, θ)
            @test cumhazard(d, t) ≈ (t / θ)^α rtol = 1e-12
        end
    end

    @testset "ForwardDiff: cumhazard at t > 0" begin
        # Gradient of cumhazard w.r.t. Weibull (shape α, scale θ) at positive t.
        t = 1.5
        f(p) = cumhazard(Weibull(p[1], p[2]), t)
        g = ForwardDiff.gradient(f, [1.3, 0.8])
        # H = (t/θ)^α; ∂α = (t/θ)^α · log(t/θ); ∂θ = -α (t/θ)^α / θ.
        α, θ = 1.3, 0.8
        @test g[1] ≈ (t / θ)^α * log(t / θ) rtol = 1e-10
        @test g[2] ≈ -α * (t / θ)^α / θ rtol = 1e-10
    end

    @testset "ForwardDiff: cumhazard at t = 0 has zero gradient" begin
        # The guard short-circuits the 0^shape path. Without it, the dual
        # value is 0 but the partials NaN through exp(α log 0).
        for d_ctor in (p -> Weibull(p[1], p[2]),
                       p -> Gamma(p[1], p[2]))
            f(p) = cumhazard(d_ctor(p), 0.0)
            g = ForwardDiff.gradient(f, [1.5, 2.0])
            @test all(iszero, g)
        end
    end

    @testset "expected_cumhazard: closed forms" begin
        # Exponential cumhaz × Exponential lifetime:
        # H_d1(t) = t/θ1; E[t] = θ2; result = θ2/θ1.
        θ1, θ2 = 0.5, 2.0
        @test expected_cumhazard(Exponential(θ1), Exponential(θ2)) ≈ θ2 / θ1 atol = 1e-8

        # Weibull(α=2, θ=1) × Exponential(scale=1):
        # H(t) = t²; E[t²] = 2 for Exp(1) (Var + mean² = 1 + 1).
        @test expected_cumhazard(Weibull(2.0, 1.0), Exponential(1.0)) ≈ 2.0 atol = 1e-6

        # Weibull α=1 reduces to Exponential.
        d_weib = Weibull(1.0, 1 / 0.5)   # rate 0.5
        d_exp = Exponential(1 / 0.5)
        @test expected_cumhazard(d_weib, Exponential(2.0)) ≈
              expected_cumhazard(d_exp, Exponential(2.0)) rtol = 1e-10
    end

    # Closed-form specialisations exist for the common survival families.
    # The tests check (a) they agree with the generic recipe at moderate t,
    # and (b) they stay finite where the generic pdf/ccdf would underflow.
    @testset "closed-form specialisations: agreement with generic" begin
        cases = (
            (Exponential(0.7),      1.0),
            (Exponential(2.0),      0.5),
            (Weibull(1.7, 0.5),     0.6),
            (Weibull(2.5, 1.0),     1.2),
            (Weibull(0.6, 2.0),     1.0),
            (Rayleigh(0.5),         0.8),
            (Rayleigh(2.0),         2.5),
            (Pareto(2.0, 1.0),      1.5),
            (Pareto(0.8, 0.5),      2.0),
            (Frechet(2.0, 1.0),     1.5),
            (Frechet(1.3, 0.5),     0.8),
        )
        for (d, t) in cases
            # Closed-form hazard agrees with the textbook pdf/ccdf recipe at
            # moderate t (where the recipe is still finite).
            @test hazard(d, t) ≈
                  Distributions.pdf(d, t) / Distributions.ccdf(d, t) rtol = 1e-10
            # Closed-form cumhazard agrees with -logccdf.
            @test cumhazard(d, t) ≈ -Distributions.logccdf(d, t) rtol = 1e-10
            # log_hazard agrees with logpdf - logccdf.
            @test log_hazard(d, t) ≈
                  Distributions.logpdf(d, t) - Distributions.logccdf(d, t) rtol = 1e-10
            # And exp(log_hazard) round-trips back to hazard.
            @test exp(log_hazard(d, t)) ≈ hazard(d, t) rtol = 1e-10
        end
    end

    @testset "closed-form specialisations: stable at extreme t" begin
        # Weibull(α=2, θ=1) at t = 100: pdf ≈ ccdf ≈ exp(-10000), both
        # underflow to 0, so the textbook pdf/ccdf ratio gives NaN. The
        # closed form is (α/θ)(t/θ)^(α-1) = 2·100 = 200.
        d = Weibull(2.0, 1.0)
        t = 100.0
        @test Distributions.pdf(d, t) == 0          # underflow
        @test Distributions.ccdf(d, t) == 0         # underflow
        # Hazard via the closed form survives.
        @test isfinite(hazard(d, t))
        @test hazard(d, t) ≈ 200.0
        # The generic exp(log_hazard) form also survives because it works
        # in log space all the way until the final exp.
        @test isfinite(exp(Distributions.logpdf(d, t) -
                           Distributions.logccdf(d, t)))

        # Frechet at large t: ccdf = -expm1(-z) ≈ z for small z. The
        # closed-form hazard is α/t to leading order.
        d = Frechet(2.0, 1.0)
        for t in (50.0, 200.0)
            z = (t / d.θ)^(-d.α)
            @test isfinite(hazard(d, t))
            @test hazard(d, t) ≈ d.α / t rtol = 0.01    # leading-order asymptote
        end
    end

    # CensoredDistributions.jl from the epiaware ecosystem composes censoring
    # on top of an underlying Distributions.jl distribution. Its types subtype
    # `Distributions.UnivariateDistribution`, so the methods here dispatch on
    # them with no further code. This testset both checks the dispatch and
    # demonstrates the integration.
    @testset "CensoredDistributions integration" begin
        d_uncensored = Gamma(2.0, 1.5)
        d = primary_censored(d_uncensored, Uniform(0.0, 1.0))

        @test d isa Distributions.UnivariateDistribution

        for t in (0.5, 1.5, 3.0)
            # The four primitives evaluate and stay consistent.
            @test cumhazard(d, t) ≈ -log_survival(d, t)
            @test density(d, t) ≈ hazard(d, t) * survival(d, t) atol = 1e-12
            @test log_density(d, t) ≈ log_hazard(d, t) + log_survival(d, t) atol = 1e-10
            # Censoring widens the survival relative to the uncensored base.
            @test survival(d, t) > 0 && survival(d, t) < 1
        end

        @test cumhazard(d, 0.0) == 0
        @test survival(d, 0.0) == 1
    end
end
