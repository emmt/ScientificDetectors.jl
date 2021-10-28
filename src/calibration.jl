module Calibration

using StatsBase, Statistics, LinearAlgebra
using SimpleExpressions
using ArrayTools
using OptimPackNextGen
using MultivariateOnlineStatistics
using MultivariateOnlineStatistics:
    storage
using ..ScientificDetectors
using ..ScientificDetectors:
    DetectorAxisTypes,
    OnlineStatistics,
    binning,
    offset
import ..ScientificDetectors:
    DetectorAxes,
    argument_error,
    dimension_mismatch,
    exposuretime
import Base: push!, merge!

const Category = Union{AbstractString,Symbol}

const Colons{N} = NTuple{N,Colon}

# Union of acceptable identifer types.
const Identifiers = Union{AbstractString,Symbol,Integer}

# Include code for types with constructors and basic method.
include("ReducedCalibration.jl")
include("CalibrationDataFrame.jl")
#include("CalibrationDataFrameSampler.jl")
include("CalibrationData.jl")
include("CalibrationFrameSampler.jl")
include("SimpleCalibration.jl")

"""
    identifier(key) -> str

converts `key` into a string identifier.  Argument `key` can be of any type part
of the union `Identifiers` (a string, a symbol or an integer).

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)

@doc @doc(identifier) Identifiers


"""
    find(obj, key) -> j

yields the index `j` of the current term in reduced calibration data which match `key`
or `0` if not found.

"""
find(obj::ReducedCalibration, key::Nothing) = 0

function find(obj::ReducedCalibration, key::AbstractString)
    cat = categories(obj)
    n = 0
    j = 0
    for i in 1:length(cat)
        if cat[i] == key
            j = i
            n += 1
        end
    end
    n > 1 && error("non-unique category identifier")
    return j
end

find(obj::ReducedCalibration, j::Integer) =
    (1 ≤ j ≤ length(categories(obj)) ? Int(j) : 0)

# Same as ArrayTools.promote_eltype but for a vector of arrays.  Using a
# recursion is the fastest method.
function _promote_eltype(x::AbstractVector{<:AbstractArray})
    n = length(x)
    @assert n ≥ 1
    return _promote_eltype((@inbounds eltype(x[n])), x, n - 1)
end
_promote_eltype(T::Type, x::AbstractVector{<:AbstractArray}, n::Int) =
    (n < 1 ? T :
     _promote_eltype(promote_type(T, (@inbounds eltype(x[n]))), x, n - 1))

#------------------------------------------------------------------------------

"""
    uniquecategories(A) -> cat, uid

given a vector `A` of identifiers or keys, yields the corresponding category
indices `cat` and unique identifers `uid` such that `uid[cat[i]]` is the unique
identifier corresponding to `A[i]`.  The elements of `A` can be of any type
belonging to the union `Identifiers` (strings, symbols or integers).

"""
function uniquecategories(A::AbstractVector{K}) where {K<:Identifiers}
    # Use a dictionary to collect a unique list of keys and then to store the
    # corresponding unique type number.
    dict = Dict{K,Int}()
    for key in A
        dict[key] = 1
    end
    l = 0
    for key in sort(collect(keys(dict)))
        l += 1
        dict[key] = l
    end

    cat = Vector{Int}(undef, length(A))
    uid = Vector{String}(undef, length(dict))
    i = 0
    for key in A
        i += 1
        cat[i] = dict[key]
    end
    for key in keys(dict)
        uid[dict[key]] = identifier(key)
    end
    return cat, uid
end

# Structure used to store the parameters of a single pixel.
mutable struct FitResult{T}
    f::T         # figure of merit
    z::T         # bias
    g::T         # gain
    u::T         # variance of readout-noise divided by gain
    s::Vector{T} # contributions of the different sources
end


# FIXME: rename as CalibrationDataFrameSampler
# FIXME: move inner constructor outside?

"""
    ReducedCalibration(cal) -> redcal

fit the detector parameters in calibration data `cal`.

"""
function ReducedCalibration(cal::CalibrationData{T,N}) where {T,N}
    nframes = numberofdataframes(cal)
    ntypes = numberofcategories(cal)
    dat = dataframes(cal)
    cat = categories(cal)
    Δt = exposuretimes(cal)
    uid = uniqueidentifiers(cal)
    dims = size(cal)
    @assert length(uid) == ntypes
    @assert length(Δt) == nframes
    @assert length(dat) == nframes
    @assert length(cat) == nframes

    # Check exposure times.
    flag = false
    for i in 1:nframes
        if !isfinite(Δt[i]) || Δt[i] < 0
            error("invalid exposure time(s)")
        end
        if Δt[i] > 0
            flag = true
        end
    end
    if !flag
        error("no non-zero exposure times!")
    end

    # Allocate output and workspaces.
    out = ReducedCalibration(
        Array{T,N}(undef, dims), # f
        Array{T,N}(undef, dims), # z
        Array{T,N}(undef, dims), # g
        Array{T,N}(undef, dims), # σ
        [Array{T,N}(undef, dims) for k in 1:ntypes], # c
        uid)
    d = Array{T,1}(undef, nframes)
    res = FitResult{T}(Inf,NaN,NaN,NaN,
                       fill!(Array{T}(undef, ntypes), NaN));

    # Fit every pixel.
    len = prod(dims)
    for j in 1:len
        # Collect the pixel data.
        for i in 1:nframes
            d[i] = cal.data[i][j]
        end

        # Fit the detector parameters and save them.
        fit!(res, d, cat, Δt)
        for l in 1:ntypes
            out.c[l][j] = res.c[l]
        end
        out.f[j] = res.f
        out.z[j] = res.z
        out.g[j] = res.g
        out.σ[j] = sqrt(res.u/res.g)
    end

    # Return reduced calibration data.
    return out
end

function fit!(res::FitResult{T}, d::Vector{T},
              cat::Vector{Int}, Δt::Vector{T};
              umin::Real = 1e-20) where {T<:AbstractFloat}
    ntypes = length(res.c)
    nframes = length(d)
    @assert length(cat) == nframes
    @assert length(Δt) == nframes

    # Initial weights are 1/(Δt + τ) with τ a small value.
    τ = leastpositive(Δt)/10
    τ > 0 || error("no non-zero exposure times")
    w = Array{T}(undef, nframes)
    update_w!(w, Δt, τ)

    # Initial bias.
    z = minimum(d)

    # Initial current terms are given by a simple constrained linear
    # least-squares fit.
    a = fill!(Array{T,1}(undef, ntypes), 0)
    b = fill!(Array{T,1}(undef, ntypes), 0)
    @inbounds for i in 1:nframes
        l = cat[i]
        a[l] += w[i]*Δt[i]^2
        b[l] += w[i]*Δt[i]*(d[i] - z)
    end
    c = Array{T,1}(undef, ntypes)
    @inbounds for l in 1:ntypes
        c[l] = (b[l] > 0 ? b[l]/a[l] : zero(T))
    end

    # Initial value of u ≡ g⋅σ² is a strictly positive value which is small
    # compared to c⋅Δt.
    cΔt = Array{T}(undef, nframes)
    update_cΔt!(cΔt, c, Δt, cat)
    u = leastpositive(cΔt)/10

    # Initialize initial variables and bounds.
    x = Array{T}(undef, ntypes+1)
    xmin = Array{T}(undef, ntypes+1)
    @inbounds for l in 1:ntypes
        x[l] = c[l]
        xmin[l] = zero(T)
    end
    x[end] = u
    xmin[end] = umin

    # Initialize result so as to store best solution so far.
    res.f = Inf
    res.z = NaN
    res.g = NaN
    res.u = NaN
    fill!(res.c, NaN)

    # Allocate workspaces r for the residuals.
    r = Array{T}(undef, nframes)

    # Define the objective function as a closure to share workspaces and data.
    function fg!(x::Vector{T}, gx::Vector{T}) where {T<:AbstractFloat}
        # Extract parameters.
        @assert length(x) == length(gx) == ntypes + 1
        @inbounds for l in 1:ntypes
            c[l] = x[l]
        end
        u = x[end]

        # Compute the contributions c⋅Δt, the weights w, the best bias z, the
        # residuals r and the best gain g.
        update_cΔt!(cΔt, c, Δt, cat, true)
        update_w!(w, cΔt, u)
        z = best_bias(w, d, cΔt)
        update_r!(r, d, cΔt, z)
        g = best_gain(w, r)

        # Compute the objective function.
        fx = zero(T)
        @inbounds @simd for i in 1:nframes
            # (5 ops + 1 log)/frames ~ 28 ops/frames
            fx += g*w[i]*r[i]^2 - log(w[i])
        end
        fx -= nframes*log(g)

        # Maybe update the best solution so far.
        if fx < res.f
            res.f = fx
            res.z = z
            res.g = g
            res.u = u
            copyto!(res.s, s)
        end

        # Compute the gradient of the objective function with respect to c.
        @inbounds for l in 1:ntypes
            gx[l] = zero(T)
        end
        @inbounds for i in 1:nframes
            # 8 ops/frames
            l = cat[i]
            gx[l] += w[i]*(1 - g*r[i]*(2 + w[i]*r[i]))*Δt[i]
        end

        # Compute the gradient of the objective function with respect to u.
        gu = zero(T)
        @inbounds @simd for i in 1:nframes
            # 6 ops/frames
            gu += w[i]*(1 - g*w[i]*r[i]^2)
        end
        gx[end] = gu

        # Return the objective function.
        return fx
    end

    # FIXME: stopping criterion
    vmlmb!(fg!, x, mem=5, lower=xmin)
end

struct FitWorkspace{T<:AbstractFloat}
    H::Matrix{T}   # sources to currents matrix
    c::Vector{T}   # temporary workspace for current terms
    ∂c::Vector{T}  # temporary workspace for gradients w.r.t. current terms
    Δt::Vector{T}  # exposure times
    l::Vector{Int} # indices of categories
    n::Vector{Int} # number of samples in subset
    avg::Vector{T} # empirical subset mean
    var::Vector{T} # empirical (biased) subset variance
end

"""
    wrk = FitWorkspace{T=float(eltype(H))}(H, nsub)

yields a workspace for fitting the parameters of the model of a detector pixel
for sources to currents matrix `H` and `nsub` data subsets.  All entries of `H`
must be nonnegative (this is checked).  The same workspace can be re-used for
another pixel provided the matrix `H` remains the same.

"""
FitWorkspace(H::AbstractMatrix{<:Real}, nobs::Integer) =
    FitWorkspace{float(eltype(H))}(H, nsub)

function FitWorkspace{T}(H::AbstractMatrix{<:Real},
                         nobs::Integer) where {T<:AbstractFloat}
    isnonnegative(H) || error(
        "entries of sources to currents matrix must be nonnegative")
    ncat, nsrc = size(H)
    return FitWorkspace{T}(
        H,
        Vector{T}(  undef, ncat), # c
        Vector{T}(  undef, ncat), # ∂c
        Vector{T}(  undef, nobs), # Δt
        Vector{Int}(undef, nobs), # l
        Vector{Int}(undef, nobs), # n
        Vector{T}(  undef, nobs), # avg
        Vector{T}(  undef, nobs)) # var
end

"""
    wrk = FitWorkspace{T=eltype(cal)}(cal)

yields a workspace for fitting the parameters of the model of a detector pixel
in calibration data `cal`.  Parameter `T` is the floating-point type for
computations.

"""
FitWorkspace(cal::CalibrationData) = FitWorkspace{eltype(cal)}(cal)
FitWorkspace{T}(cal::CalibrationData) where {T<:AbstractFloat} =
    FitWorkspace{T}(cal.src_to_cat, length(cal.stat))

# Return the number of sufficient data in fit workspace.
function checked_length(wrk::FitWorkspace; checkindices::Bool=false)
    ncat, nsrc = size(wrk.H)
    lenght(wrk.c) == ncat || error("invalid number of current terms")
    lenght(wrk.∂c) == ncat || error("invalid number of gradients w.r.t. current terms")
    if checkindices
        flag = true
        @inbounds @simd for i in eachindex(wrk.l)
            flag &= ((wrk.l[i] - 1)%UInt < ncat)
        end
    flag || error("out of bound category index")
    end
    nobs = length(wrk.l)
    length(wrk.Δt) == nobs || error("bad number of exposure times")
    length(wrk.avg) == nobs || error("bad number of empirical means")
    length(wrk.var) == nobs || error("bad number of empirical variances")
    length(wrk.n) == nobs || error("bad number of subset sizes")
    return nobs
end

"""
    extract!(wrk, cal, k) -> wrk

extracts into workspace `wrk` the data for `k`-th pixel in calibration data `cal`.

"""
function extract!(wrk::FitWorkspace,
                  cal::CalibrationData{T,N},
                  k::Integer) where {T,N}
    avg = cal.stat
    ncat, nsrc = size(cal.src_to_cat)
    nsub = length(cal.stat)
    nsub == checked_length(wrk) || error(
        "fit workspace assumes a different number of subsets")
    for (key, i) ∈ cal.stat_index
        # Extract (Δt,ℓ) for the subset of calibration data.
        cat       = key[1]             # category name
        wrk.Δt[i] = key[2]             # exposure time
        wrk.l[i]  = cal.cat_index[cat] # category index

        # Extract statistics of k-th pixel in subset of calibration data
        # samples.
        stat       = cal.stat[i]
        wrk.n[i]   = nobs(stat)
        wrk.avg[i] = mean(stat, k)
        wrk.var[i] = var(stat, k; corrected=false)
    end
    return wrk
end

struct NormalEquations{T<:AbstractFloat,
                       LHS<:AbstractMatrix{T},
                       RHS<:AbstractVector{T}}
    A::LHS
    b::RHS
    function NormalEquations{T}(A::LHS, b::RHS) where {T<:AbstractFloat,
                                                       LHS<:AbstractMatrix{T},
                                                       RHS<:AbstractVector{T}}
        I, J = axes(A)
        I == J || argument_error(
            "expecting a square LHS matrix")
        axes(b) == (I,) || argument_error(
            "LHS matrix and RHS vector have incompatible indices")
        return new{T,LHS,RHS}(A, b)
    end
end

lhs(E::NormalEquations) = getfield(E, :A)
rhs(E::NormalEquations) = getfield(E, :b)

function NormalEquations(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    T = float(eltype(A), eltype(b))
    return NormalEquations{T}(A, b)
end

function NormalEquations{T}(A::AbstractMatrix{<:Real},
                            b::AbstractVector{<:Real}) where {T<:AbstractFloat}
    return NormalEquations{T}(convert(AbstractMatrix{T}, A),
                              convert(AbstractVector{T}, b))
end

# Objective function and its gradient:
#   f(x) = (1/2) x'⋅A⋅x - b⋅x
#   ∇f(x) = A⋅x - b
#   =>  f(x) = (1/2) x'⋅(∇f(x) - b)
function (E::NormalEquations)(x::AbstractVector{T}) where {T<:AbstractFloat}
    A, b = lhs(E), rhs(E)
    I = axes(b)
    axes(A) == (I,I) || error(
        "LHS matrix and RHS vector have incompatible indices")
    axes(x) == (I,) || error(
        "input variables have incompatible indices")
    f = zero(T)
    @inbounds for i in I
        # Compute A⋅x = A'⋅x (A is symmetric)
        Ax = zero(T)
        @simd for j in I
            Ax += A[j,i]*x[j]
        end
        f += (Ax - 2b[i])*x[i]
    end
    return f/2
end

function (E::NormalEquations)(x::AbstractVector{T},
                              g::AbstractVector{T}) where {T<:AbstractFloat}
    A, b = lhs(E), rhs(E)
    I = axes(b)
    axes(A) == (I,I) || error(
        "LHS matrix and RHS vector have incompatible indices")
    axes(x) == (I,) || error(
        "input variables have incompatible indices")
    axes(g) == (I,) || error(
        "output gradient has incompatible indices")
    f = zero(T)
    @inbounds for i in I
        # Compute A⋅x = A'⋅x (A is symmetric)
        Ax = zero(T)
        @simd for j in I
            Ax += A[j,i]*x[j]
        end
        bi = b[i]
        gi = Ax - bi
        g[i] = gi
        f += (gi - bi)*x[i]
    end
    return f/2
end

"""
    fit_linear_terms!(wrk, x; eta=Inf, kwds...)

fits the linear terms of the pixel detector model (the bias `z` and the source
terms).

Other keywords are passed to `vmlmb!`.

"""
function fit_linear_terms!(wrk::FitWorkspace{T},
                           x::Vector{T};
                           # FIXME: add option to re-weight
                           eta::Real = Inf,
                           mem::Integer = 5,
                           kwds...) where {T<:AbstractFloat}
    # Do a weighted least squares fit on all the linear parameters with the
    # positivity constraint on the source terms.
    H = wrk.H
    ncat, nsrc = size(H)
    nsub = checked_length(wrk; checkindices=true)

    # Integrate W1, W2, and W3 over the categories.
    w1 = fill!(Vector{T}(undef, ncat), zero(T)) # FIXME: make is part of wrk
    w2 = fill!(Vector{T}(undef, ncat), zero(T)) # FIXME: make is part of wrk
    w3 = fill!(Vector{T}(undef, ncat), zero(T)) # FIXME: make is part of wrk
    Ann = zero(T)
    bn = zero(T)
    if eta == Inf
        for i in 1:nsub
            Δt = wrk.Δt[i]
            l = wrk.l[i]
            w = T(wrk.n[i])
            d = wrk.avg[i]
            wΔt = w*Δt
            w1[l] += wΔt
            w2[l] += wΔt*Δt
            w3[l] += wΔt*d
            Ann += w
            bn += w*d
        end
    else
        eta > 0 || argument_error("value of keyword `eta` must be positive")
        c = mvmult!(wrk.c, H, view(x, 1:nsrc))
        η = to_type(T, eta)
        for i in 1:nsub
            Δt = wrk.Δt[i]
            l = wrk.l[i]
            cΔt = c[l]*Δt
            w = wrk.n[i]/(cΔt + η)
            d = wrk.avg[i]
            wΔt = w*Δt
            w1[l] += wΔt
            w2[l] += wΔt*Δt
            w3[l] += wΔt*d
            Ann += w
            bn += w*d
        end
    end

    # Compute A the LHS matrix of the normal equations.
    m, n = ncat, nsrc + 1
    A = Matrix{T}(undef, n, n) # FIXME: make is part of wrk
    u = Vector{T}(undef, m) # FIXME: make is part of wrk
    for j ∈ 1:n-1
        # First (n-1)×(n-1) block.
        for l ∈ 1:m
            u[l] = H[l,j]*w2[l]
        end
        for jp ∈ 1:n-1
            s = zero(T)
            for l ∈ 1:m
                s += u[l]*H[l,jp]
            end
            A[j,jp] = s
            if jp == j
                break
            end
            A[jp,j] = s
        end
        # Last column and last row of A.
        s = zero(T)
        for l ∈ 1:m
            s += H[j,l]*w1[l]
        end
        A[j,n] = s
        A[n,j] = s
    end
    A[n,n] = Ann

    # Compute b the RHS vector of the normal equations.
    b = Vector{T}(undef, n) # FIXME: make is part of wrk
    for j ∈ 1:n-1
        s = zero(T)
        for l ∈ 1:m
            s += H[j,l]*w3[l]
        end
        b[j] = s
    end
    b[n] = bn

    # Solve the normal equations under the constraints that the source terms
    # are nonnegative.
    E = NormalEquations{T}(A, b)
    x = Vector{T}(undef, n)
    xmin = Vector{T}(undef, n)
    @inbounds for i in 1:n-1
        xmin[i] = 0 # source terms are nonnegative
    end
    xmin[n] = -Inf # z is unbounded
    fill!(x, 0)
    vmlmb!(E, x; mem=mem, lower=xmin, kwds...)
    return x

    #x = Vector{T}(undef, nsrc + 1)
    #xmin = Vector{T}(undef, nsrc + 1)
    #xmin[1] = -Inf # z is unbounded
    #@inbounds for i in 2:nsrc+1
    #    xmin[i] = 0
    #end
    #fill!(x, 0)
    #function fg!(x::Vector{T}, gx::Vector{T}) where {T<:AbstractFloat}
    #    n = length(x)
    #    z = x[1]
    #    c = mvmult!(wrk.c, wrk.H, view(x, 2:n))
    #    ∂c = fill!(wrk.∂c, zero(T))
    #    ∂z = zero(T)
    #    f = zero(T)
    #    for i in 1:nsub
    #        Δt = wrk.Δt[i]
    #        l = wrk.l[i]
    #        w = T(wrk.n[i])
    #        r = (c[l]*Δt + z) - wrk.avg[i]
    #        wr = w*r
    #        f += wr*r
    #        ∂c[l] += wr*Δt
    #        ∂z += wr
    #    end
    #    # convert gradient and return objective function
    #    gx[1] = ∂z
    #    mvmult!(view(gx, 2:n), wrk.H', ∂c)
    #    return f/2
    #end
end


"""
    f = objfunc(wrk, z, g, η, s)

yields the value of the objective function associated with workspace `wrk` and
for model parameters `x = (z, g, η, s...)` with `z` the zero level, `g` the
gain, `η` the variance of the readout noise times the gain and the source terms
`s`.

"""
function objfunc(wrk::FitWorkspace{T}, z::Real, g::Real, η::Real,
                 s::AbstractVector{T}) where {T<:AbstractFloat}
    return objfunc(wrk, to_type(T, z), to_type(T, g), to_type(T, η), s)
end

function objfunc(wrk::FitWorkspace{T}, z::T, g::T, η::T,
                 s::AbstractVector{T}) where {T<:AbstractFloat}
    g > 0 || throw_argument_error("gain `g` must be positive")
    η > 0 || throw_argument_error("readout variance `η` must be positive")
    isnonnegative(s) || throw_argument_error("source terms `s` must be nonnegative")
    c = mvmult!(wrk.c, wrk.H, s)
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    @inbounds for i ∈ 1:checked_length(wrk; checkindices=true)
        Δt = wrk.Δt[i]  # exposure time
        l = wrk.l[i]    # category index
        n = T(wrk.n[i]) # number of samples in subset
        cΔt = c[l]*Δt   # contribution of sources
        r = (cΔt + z) - wrk.avg[i] # residuals: model - sample mean
        w = n/(cΔt + η) # weight
        χ² += w*(wrk.var[i] + r^2)
        sum_n_logw_n += n*log(w/n)
        N += n
    end
    return g*χ² - sum_n_logw_n - N*log(g)
end

"""
    f = objfunc!(wrk, z, g, η, s, grd)

yields the value of the objective function `f(x)` associated with workspace
`wrk` and overwrites `grd` with the gradient `∇f(x)` for model parameters `x =
(z, g, η, s...)` with `z` the zero level, `g` the gain, `η` the variance of the
readout noise times the gain and the source terms `s`.

"""
function objfunc!(wrk::FitWorkspace{T},
                  z::Real, g::Real, η::Real, s::AbstractVector{T},
                  grd::AbstractVector) where {T<:AbstractFloat}
    return objfunc!(wrk, to_type(T, z)::T, to_type(T, g)::T, to_type(T, η)::T,
                    s, grd)
end

function objfunc!(wrk::FitWorkspace{T},
                  z::T, g::T, η::T, s::AbstractVector{T},
                  grd::AbstractVector) where {T<:AbstractFloat}
    g > 0 || throw_argument_error("gain `g` must be positive")
    η > 0 || throw_argument_error("readout variance `η` must be positive")
    isnonnegative(s) ≥ 0 || throw_argument_error("source terms `s` must be nonnegative")
    c = mvmult!(wrk.c, wrk.H, s) # compute current terms
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    ∂c = fill!(wrk.∂c, 0) # to compute ∂L/∂c
    ∂z = zero(T) # to compute ∂L/∂z
    ∂η = zero(T) # to compute ∂L/∂η
    @inbounds for i ∈ 1:checked_length(wrk; checkindices=true)
        Δt = wrk.Δt[i]  # exposure time
        l = wrk.l[i]    # category index
        n = T(wrk.n[i]) # number of samples in subset
        cΔt = c[l]*Δt   # contribution of sources
        r = (cΔt + z) - wrk.avg[i] # residuals: model - sample mean
        q = cΔt + η
        w = n/(cΔt + η) # weight
        v = wrk.var[i] + r^2
        χ² += w*v
        w_n = w/n
        sum_n_logw_n += n*log(w_n)
        N += n
        ∂m = 2*g*w*r # ∂L/∂m
        ∂w = g*v - q # ∂L/∂w
        ρ = w*w_n*∂w # (w²/n)⋅(∂L/∂w)
        ∂z += ∂m
        ∂η -= ρ
        ∂c[l] += (∂m - ρ)*Δt
    end
    ∂g = χ² - N/g # ∂L/∂g
    # Store/convert gradients.
    grd[1] = ∂z
    grd[2] = ∂g
    grd[3] = ∂η
    mvmult!(view(grd, 4:length(grd)), wrk.H', ∂c)
    # Return total objective function.
    return g*χ² - sum_n_logw_n - N*log(g)
end

"""
    mvmult!(y, A, x) -> y

overwrites destination vector `y` with the matrix-vector product `A⋅x` and
returns `y`.  This method is similar to `LinearAlgebra.lmul!` but does not call
any BLAS subroutine.

"""
function mvmult!(y::AbstractVector,
                 A::AbstractMatrix{Ta},
                 x::AbstractVector) where {Ta}
    I, J = axes(A)
    axes(y) == (I,) || throw_dimension_mismatch(
        "incompatible indices of destination vector")
    axes(x) == (J,) || throw_dimension_mismatch(
        "incompatible indices of source vector")
    T = promote_type(Ta, eltype(x))
    @inbounds for i ∈ I
        s = zero(T)
        @simd for j ∈ J
            s += A[i,j]*x[j]
        end
        y[i] = s
    end
    return y
end

function mvmult!(y::AbstractVector,
                 A′::Adjoint{Ta,<:AbstractMatrix{Ta}},
                 x::AbstractVector) where {Ta}
    A = parent(A′)
    I, J = axes(A)
    axes(y) == (J,) || throw_dimension_mismatch(
        "incompatible indices of destination vector")
    axes(x) == (I,) || throw_dimension_mismatch(
        "incompatible indices of source vector")
    T = promote_type(Ta, eltype(x))
    @inbounds for j ∈ J
        s = zero(T)
        @simd for i ∈ I
            s += conj(A[i,j])*x[i]
        end
        y[j] = s
    end
    return y
end

"""
    isnonnegative(A)

yields whether `A` is nonnegative.

"""
isnonnegative(x::Real) = (x ≥ zero(x))
function isnonnegative(A::AbstractArray{<:Real})
    # Purposely use non-branching expression as we are mostly interested in
    # cases for which the result is true and thus all entries have to be
    # checked so short-circuit is not an advantage.
    flag = true
    @inbounds @simd for i in eachindex(A)
        flag &= isnonnegative(A[i])
    end
    return flag
end

"""
    to_type(T, x)

yields `x` converted to type `T`, the result is asserted to be of type `T`.

"""
to_type(::Type{T}, x::T) where {T} = x
to_type(::Type{T}, x::Any) where {T} = convert(T, x)::T

"""
    checkindices(I, len)

checks that all indices in `I` are in the range `1:len`.  An error is thrown
if `len ≤ 0` of if any values in `I` is outside the range `1:len`.

"""
function checkindices(I::AbstractArray{U}, len::Integer) where {U<:Unsigned}
    len > 0 || error("invalid length")
    if len < typemax(U)
        lim = U(len)
        @inbounds for i in I
            i - one(U) < lim || error("out of bound type index")
        end
    end
end

function checkindices(I::AbstractArray{S}, len::Integer) where {S<:Signed}
    len > 0 || error("invalid length")
    if len < typemax(S)
        U = unsigned(S)
        lim = U(len)
        @inbounds for i in I
            (i % U) - one(U) < lim || error("out of bound type index")
        end
    else
        # Just check for sign.
        @inbounds for i in I
            i > zero(U) || error("out of bound type index")
        end
    end
end

"""
    update_w!(w, cΔt, u) -> w

overwrites `w` with `1/(c⋅Δt + u)`, that is do `∀i: w[i] = 1/(cΔt[i] + u)`, and
returns `w`.

See also [`update_cΔt!`](@ref).

"""
function update_w!(w::Vector{T}, cΔt::Vector{T}, u::T) where {T<:AbstractFloat}
    u′ = T(u)
    @inbounds @simd for i ∈ eachindex(w, cΔt)
       w[i] = one(T)/(cΔt[i] + u′)
    end
    return w
end

"""
    update_cΔt!(cΔt, c, Δt, cat, nochecks=false) -> cΔt

overwrites `cΔt` with `c⋅Δt`, that is do `∀i: cΔt[i] = c[cat[i]]*Δt[i]`, and
returns `cΔt`.  Set optional argument `nochecks` to `true` to skip testing
the indices in `cat`.

See also  [`update_w!`](@ref), [`checkindices`](@ref).

"""
function update_cΔt!(cΔt::Vector{T}, c::Vector{T},
                     Δt::Vector{T}, cat::Vector{Int},
                     nochecks::Bool = false) where {T<:AbstractFloat}
    nframes = length(cΔt)
    @assert length(Δt) == nframes
    @assert length(cat) == nframes
    nochecks || checkindices(cat, length(c))
    @inbounds for i ∈ 1:nframes
        cΔt[i] = c[cat[i]]*Δt[i]
    end
    return cΔt
end


"""
    update_r!(r, d, cΔt, z) -> r

overwrites array `r` with the residuals given the data `d`, the contribution
`cΔt` of the different sources and the bias `z`. The destination `r` is
returned.  The residuals are computed as:

    ∀i: r[i] = d[i] - cΔt[i] - z

See also [`update_cΔt!`](@ref), [`best_bias`](@ref).

"""
function update_r!(r::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T},
                   z::Real) where {T<:AbstractFloat}
    z′ = T(z)
    @inbounds @simd for i ∈ eachindex(r, d, cΔt)
        r[i] = d[i] - (cΔt[i] + z′)
    end
    return r
end

"""
    best_bias(w, d, cΔt) -> z

yields the best bias given the weights `w`, the data `d` and the contribution
`cΔt` of the different sources.

See also [`update_w!`](@ref), [`update_cΔt!`](@ref), [`best_gain`](@ref).

"""
function best_bias(w::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T}) where {T<:AbstractFloat}
    a, b = zero(T), zero(T)
    @inbounds @simd for i ∈ eachindex(w, d, cΔt)
        a += w[i]
        b += w[i]*(d[i] - cΔt[i])
    end
    return b/a
end

"""
    best_gain(w, d, cΔt, z) -> g

yields the best gain given the weights `w`, the data `d`, the contribution `cΔt`
of the different sources and the bias `z`.

Alternatively, if the residuals `r = d - cΔt - z` have been computed, just
call:

    best_gain(w, r) -> g

See also [`update_w!`](@ref), [`update_cΔt!`](@ref), [`update_r!`](@ref),
[`best_bias`](@ref).

"""
function best_gain(w::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T},
                   z::Real) where {T<:AbstractFloat}
    z′ = T(z)
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(w, d, cΔt)
        s += w[i]*(cΔt[i] + z′ - d[i])^2
    end
    return length(w)/s
end

function best_gain(w::AbstractVector{T},
                   r::AbstractVector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(w, r)
        s += w[i]*r[i]^2
    end
    return length(w)/s
end

"""
    leastpositive(A)

yields the least strictly positive value of array `A` or zero if all values of
`A` are nonpositive.

"""
function leastpositive(A::AbstractArray{T}) where {T}
    res = zero(T)
    @inbounds for val in A
        if val > zero(T) && (res > val || res == zero(T))
            res = val
        end
    end
    return res
end

end # module
