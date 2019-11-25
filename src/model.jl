module DetectorModel

using Printf
using SpecialFunctions

const PLOTTING = true
if PLOTTING
    const FIGDIR = joinpath(@__DIR__, "../tex/figs")
    using PyPlot
    const plt = PyPlot
end

# ~ 10 times faster `BigFloat(factorial(big(n)))` compared to `factorial(BigFloat(n))`.
# Lookup tables for factorials.
const FACT_INT = Array{BigInt}(undef, 0)
const FACT_FLT = Array{BigFloat}(undef, 0)

function fact(::Type{BigInt}, n::Integer)
    n == 0 && return one(BigInt)
    n < 0 && throw(ArgumentError("invalid negative value"))
    n ≤ length(FACT_INT) || grow_fact_tables(n)
    FACT_INT[n]
end

function fact(::Type{BigFloat}, n::Integer)
    n == 0 && return one(BigFloat)
    n < 0 && throw(ArgumentError("invalid negative value"))
    n ≤ length(FACT_FLT) || grow_fact_tables(n)
    FACT_FLT[n]
end

function grow_fact_tables(n::Integer)
    oldlen = min(length(FACT_INT), length(FACT_FLT))
    newlen = round(Int, sqrt(2)*n)
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
    s0 ≈ 1 || @warn "the PDF is not normalized"
    µ = s1/s0

    # Second compute the variance.
    s2 = zero(Float64)
    @inbounds @simd for i in eachindex(x, y)
        s2 += Float64(y[i]*(x[i]- µ)^2)
    end
    σ² = s2/s0

    return µ, σ²
end

function pdf(D::AbstractVector{<:Integer}, µ::Real, g::Real, z::Real)
    h = Array{Float64}(undef, length(D))
    i = 0
    mu = BigFloat(µ)
    q = exp(-mu)
    for d in D
        i += 1
        ninf = max(0, ceil(Int, (d - z - 0.5)*g))
        #nsup = max(0, ceil(Int, (d - z + 0.5)*g) - 1)
        lim = (d - z + 0.5)*g # strict upper bound
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

function plot_histograms(savefigs::Bool=false)
    mu = 200;
    x = 0:255;
    g1, z1 = 3.0,   0
    g2, z2 = 3.1,  35
    g3, z3 = 3.2,  70
    g4, z4 = 3.5, 105
    h1 = DetectorModel.pdf(x, mu, g1, z1)
    h2 = DetectorModel.pdf(x, mu, g2, z2)
    h3 = DetectorModel.pdf(x, mu, g3, z3)
    h4 = DetectorModel.pdf(x, mu, g4, z4)
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

function plot_biases(savefigs::Bool=false)
    z = 0.0
    x = 0:2000

    means = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 300.0, 1000]
    gains = [1.0, 2.0, 2.3, 3.0, 3.2, 3.4, 5.0, 6.0, 7.0, 8.1]

    res = Array{Float64}(undef, 2, length(means), length(gains))
    for j in eachindex(gains)
        g = gains[j]
        for i in eachindex(means)
            µ = means[i]
            y = pdf(x, µ, g, z)
            m, v = stat(x, y)
            res[1,i,j] = m
            res[2,i,j] = v
        end
    end
    if PLOTTING
        plt.figure(2)
        plt.clf()
        for j in eachindex(gains)
            g = gains[j]
            q = 1/g
            plt.plot(means, res[1,:,j] .- q.*means, label="gain = $g")
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

        plt.figure(3)
        plt.clf()
        for j in eachindex(gains)
            g = gains[j]
            q = 1/g^2
            plt.plot(means, res[2,:,j] .- q.*means, label="gain = $g")
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
