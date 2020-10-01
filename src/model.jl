module DetectorModel

using Printf
using SpecialFunctions

const PLOTTING = true
if PLOTTING
    const FIGDIR = joinpath(@__DIR__, "../tex/figs")
    using PyPlot
    const plt = PyPlot
end

# It is ~ 10 times faster to do `BigFloat(factorial(big(n)))` rather than
# `factorial(BigFloat(n))`.

# Lookup tables for factorials.
const FACT_INT = Array{BigInt}(undef, 0)
const FACT_FLT = Array{BigFloat}(undef, 0)

"""
```julia
fact(T, n) -> n!
```

yields the value of `n!` for integer `n ≥ 0`.  Argument `T` (`BigFloat` or
`BigInt`) is the type of the result.  Using a type able to represent large
numbers is mandatory as `n!` quickly grows with `n`.  An internal look-up table
is used to speed up computations.  As a result `fact(n)` is much faster than
`factorial(n)` if `n!` is needed more than once for the same value of `n` or
for close values of `n`.

The method `grow_factorial_tables(n)` may be called to pre-compute factorials
up to `n`.

Also see: [`factorial`](@ref).

See https://www.johndcook.com/blog/2010/08/16/how-to-compute-log-factorial/
for a method to compute `log(n!)` with a sufficient precision.

"""
function fact(::Type{BigInt}, n::Integer)
    n == 0 && return one(BigInt)
    n < 0 && throw(ArgumentError("invalid negative value"))
    n ≤ length(FACT_INT) || grow_factorial_tables(round(Int, sqrt(2)*n))
    return @inbounds FACT_INT[n]
end

function fact(::Type{BigFloat}, n::Integer)
    n == 0 && return one(BigFloat)
    n < 0 && throw(ArgumentError("invalid negative value"))
    n ≤ length(FACT_FLT) || grow_factorial_tables(round(Int, sqrt(2)*n))
    return @inbounds FACT_FLT[n]
end

function grow_factorial_tables(n::Integer)
    oldlen = min(length(FACT_INT), length(FACT_FLT))
    newlen = Int(n)
    if oldlen < newlen
        resize!(FACT_INT, newlen)
        resize!(FACT_FLT, newlen)
        val = (oldlen < 1 ? one(BigInt) : FACT_INT[oldlen])
        num = BigInt(oldlen)
        @inbounds for i in (oldlen+1):newlen
            num += 1
            val *= num
            FACT_INT[i] = val
            FACT_FLT[i] = BigFloat(val)
        end
    end
end

@doc @doc(fact) grow_factorial_tables

"""
```julia
stat(x, y) -> µ, σ²
```

yields the mean and the variance of the sampled distribution `y = pdf(x)`
where `pdf` denotes the Probability Density Function.

"""
function stat(x::AbstractVector{<:Integer}, y::AbstractVector{<:Real})
    @assert !Base.has_offset_axes(x, y)
    @assert length(x) == length(y)
    @assert minimum(y) ≥ 0 "the PDF must be nonnegative"

    # First compute the mean.
    s0 = zero(Float64)
    s1 = zero(Float64)
    @inbounds @simd for i in eachindex(x, y)
        s0 += Float64(y[i])
        s1 += Float64(y[i]*x[i])
    end
    if abs(s0 - 1) > 2e-6
        printstyled(stderr, "Warning: ", bold=true, color=:yellow)
        print(stderr, "The PDF is not normalized (Σpdf = $s0).\n")
    end
    µ = s1/s0

    # Second compute the variance.
    s2 = zero(Float64)
    @inbounds @simd for i in eachindex(x, y)
        s2 += Float64(y[i]*(x[i]- µ)^2)
    end
    σ² = s2/s0

    return µ, σ²
end

"""
```julia
pdf(d, µ, γ, β) -> p
```

yields the probability of measuring the ADU's in vector `d` for a mean number
of electrons `µ` with a detector gain `γ` and bias `β`.

"""
function pdf(D::AbstractVector{<:Integer}, µ::Real, γ::Real, β::Real)
    T = Float64
    mu = T(µ)
    gamma = T(γ)
    beta = T(β)
    logmu = log(mu)
    h = Array{Float64}(undef, length(D))
    i = 0
    for d in D
        i += 1
        ninf = max(0, ceil(Int, (d - beta - 0.5)*gamma))
        nsup = max(0, ceil(Int, (d - beta + 0.5)*gamma) - 1)
        if nsup ≥ ninf
            r = one(T)
            for n in nsup:-1:ninf+1
                r = one(T) + r*mu/n
            end
            h[i] = exp(ninf*logmu - mu - loggamma(ninf + 1))*r
        else
            h[i] = 0
        end
    end
    return h
end

# This version uses large number arithmetic and is typically 40 times slower
# than the fast version.
function slowpdf(D::AbstractVector{<:Integer}, µ::Real, γ::Real, β::Real)
    mu = BigFloat(µ)
    gamma = BigFloat(γ)
    beta = BigFloat(β)
    half = 1/BigFloat(2)
    q = exp(-mu)
    h = Array{Float64}(undef, length(D))
    i = 0
    for d in D
        i += 1
        ninf = max(0, ceil(Int, (d - beta - half)*gamma))
        #nsup = max(0, ceil(Int, (d - beta + half)*gamma) - 1)
        lim = (d - beta + half)*gamma # strict upper bound
        r = q*mu^ninf/fact(BigFloat, ninf)
        s = zero(BigFloat)
        n = ninf
        while n < lim
            s += r
            n += 1
            if n < lim
                r *= mu/BigFloat(n)
            end
        end
        h[i] = s
    end
    return h
end

function plot_histograms(; savefigs::Bool=false)
    mu = 200;
    x = 0:255;
    g1, z1 = 3.0,   0
    g2, z2 = 3.1,  35
    g3, z3 = 3.2,  70
    g4, z4 = 3.5, 105
    h1 = pdf(x, mu, g1, z1)
    h2 = pdf(x, mu, g2, z2)
    h3 = pdf(x, mu, g3, z3)
    h4 = pdf(x, mu, g4, z4)
    plt.figure(1)
    plt.clf()
    plt.step(x, h1, where="mid", label="\$\\mu\$ = $mu, \$g\$ = $g1, \$z\$ = $z1")
    plt.step(x, h2, where="mid", label="\$\\mu\$ = $mu, \$g\$ = $g2, \$z\$ = $z2")
    plt.step(x, h3, where="mid", label="\$\\mu\$ = $mu, \$g\$ = $g3, \$z\$ = $z3")
    plt.step(x, h4, where="mid", label="\$\\mu\$ = $mu, \$g\$ = $g4, \$z\$ = $z4")
    plt.xlabel("Data Level (ADU)")
    plt.axis(xmin=19, xmax=179, ymax=0.13)
    plt.ylabel("Probability")
    plt.legend()
    if savefigs
        plt.savefig(joinpath(FIGDIR,"histograms.pdf"), format="pdf")
    end
end

gaussian(x::AbstractVector{<:Real}, μ::Real, σ²::Real) =
    exp.(-1/(2*σ²)*(x .- μ).^2).*(1/sqrt(2*π*σ²))

function plot_approximations(; savefigs::Bool=false)
    mu = 200
    g = 3.2
    z = 0

    x1 = 0:255
    h1 = pdf(x1, mu, g, z)

    x2 = range(first(x1), last(x1), step=0.1)
    t = (x2 .- z).*g
    h2 = exp.((log(g) - mu) .+ log(mu).*t .- loggamma.(t .+ 1))

    x3 = x2
    h3 = gaussian(x3, mu/g, mu/g^2 + 1/12)

    plt.figure(2)
    plt.clf()
    plt.step(x1, h1, where="mid", label="exact")#"\$\\mu\$ = $mu, \$g\$ = $g, \$z\$ = $z")
    plt.plot(x2, h2, label="median approx.")
    plt.plot(x3, h3, label="Gaussian approx.")
    plt.xlabel("Data Level (ADU)")
    plt.axis(xmin=44, xmax=81)
    plt.ylabel("Probability")
    plt.legend()
    if savefigs
        plt.savefig(joinpath(FIGDIR,"pdf-approx.pdf"), format="pdf")
    end
end

function plot_biases(; savefigs::Bool=false, averaging::Bool=true, npts::Integer=201)

    fluxes = exp.(range(0,7,length=npts))
    #fluxes = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 1000]
    gains = [1.0, 2.0, 2.3, 3.0, 3.2, 3.4, 5.0, 6.0, 7.0, 8.1]

    res = Array{Float64}(undef, 2, length(fluxes), length(gains))
    for j in eachindex(gains)
        g = gains[j]
        for i in eachindex(fluxes)
            µ = fluxes[i]
            if averaging
                Σm = 0.0
                Σv = 0.0
                n = 100
                for k in 0:(n-1)
                    z = k/n
                    x = 0:ceil(Int, (µ/g + z) + 10*(µ/g^2 + 1/12))
                    y = pdf(x, µ, g, z)
                    m, v = stat(x, y)
                    Σm += m - z
                    Σv += v
                end
                res[1,i,j] = Σm/n
                res[2,i,j] = Σv/n
            else
                x = 0:ceil(Int,  (µ/g + z) + 10*(µ/g^2 + 1/12))
                z = 0.0
                y = pdf(x, µ, g, z)
                m, v = stat(x, y)
                res[1,i,j] = m - z
                res[2,i,j] = v
            end
        end
    end
    if PLOTTING
        plt.figure(3)
        plt.clf()
        for j in eachindex(gains)
            g = gains[j]
            q = 1/g
            plt.plot(fluxes, res[1,:,j] .- q.*fluxes, label="gain = $g")
        end
        #plt.title("Mean of detector signal")
        plt.xlabel("Flux \$\\mu\$ (electrons)")
        plt.ylabel("Mean of Signal - flux/gain")
        plt.axis(xmin=0.07)
        plt.xscale("log")
        plt.legend()
        if savefigs
            plt.savefig(joinpath(FIGDIR,"means.pdf"), format="pdf")
        end

        plt.figure(4)
        plt.clf()
        for j in eachindex(gains)
            g = gains[j]
            q = 1/g^2
            plt.plot(fluxes, res[2,:,j] .- q.*fluxes, label="gain = $g")
        end
        #plt.title("Variance of detector signal")
        plt.xlabel("Flux \$\\mu\$ (electrons)")
        plt.ylabel("Variance of Signal - flux/gain²")
        plt.axis(xmin=0.07)
        plt.xscale("log")
        plt.legend()
        if savefigs
            plt.savefig(joinpath(FIGDIR,"variances.pdf"), format="pdf")
        end
    end
end

end # module
