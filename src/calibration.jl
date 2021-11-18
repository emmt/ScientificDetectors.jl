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
# FIXME: Only needed by ReducedCalibration.
const Identifiers = Union{AbstractString,Symbol,Integer}

# Include code for types with constructors and basic method.
include("ReducedCalibration.jl")
include("CalibrationDataFrame.jl")
include("CalibrationData.jl")
include("CalibrationFrameSampler.jl")
include("SimpleCalibration.jl")

#------------------------------------------------------------------------------
# FIXME: Only needed by ReducedCalibration.
"""
    identifier(key) -> str

converts `key` into a string identifier.  Argument `key` can be of any type
part of the union `Identifiers` (a string, a symbol or an integer).

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)

@doc @doc(identifier) Identifiers

# FIXME: Only needed by ReducedCalibration.
"""
    find(obj, key) -> j

yields the index `j` of the current term in reduced calibration data which
match `key` or `0` if not found.

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
# recursion is the fastest method.  FIXME: Only needed by ReducedCalibration.
function _promote_eltype(x::AbstractVector{<:AbstractArray})
    n = length(x)
    @assert n ≥ 1
    return _promote_eltype((@inbounds eltype(x[n])), x, n - 1)
end
_promote_eltype(T::Type, x::AbstractVector{<:AbstractArray}, n::Int) =
    (n < 1 ? T :
     _promote_eltype(promote_type(T, (@inbounds eltype(x[n]))), x, n - 1))

#------------------------------------------------------------------------------

struct ObjectiveFunction{S,T<:AbstractFloat}
    H::Matrix{T}   # sources to currents matrix
    c::Vector{T}   # temporary workspace for category terms
    ∂c::Vector{T}  # temporary workspace for gradients w.r.t. current terms
    Δt::Vector{T}  # exposure times
    l::Vector{Int} # indices of categories
    n::Vector{Int} # number of samples in subset
    avg::Vector{T} # empirical subset mean
    var::Vector{T} # empirical (biased) subset variance
    w1::Vector{T}  # 1st temporary vector of same length as `c`
    w2::Vector{T}  # 2nd temporary vector of same length as `c`
    w3::Vector{T}  # 3rd temporary vector of same length as `c`
    w4::Vector{T}  # 4th temporary vector of same length as `c`
    A::Matrix{T}   # LHS matrix of the normal equations
    b::Vector{T}   # RHS vector of the normal equations
end

"""
    obj = ObjectiveFunction{S,T=float(eltype(H))}(H, nsub)

yields a workspace for fitting the parameters of the model of a detector pixel
for sources to currents matrix `H` and `nsub` data subsets.  All entries of `H`
must be nonnegative (this is checked).  The same workspace can be re-used for
another pixel provided the matrix `H` remains the same.  Type parameter `S` is
a symbol which specifies the chosen parametrization for the unknowns `θ`:

- `S = :orig` to have parameters `θ = [z, g, σ, s...]`;

- `S = :alt` to have parameters `θ = [z, g, η, s...]`;

- `S = :hier` to have parameters `θ = [η, s...]` while `z` and `g` are
  automatically derived from the others;

with `z` the zero-level (in ADU), `g` the gain (in e-/ADU), `σ` the standard
deviation of the readout noise (in ADU), `s` the source terms, and `η = g*σ^2`.

Object `obj` is callable:

    obj(θ)      # yields objective function value for parameters `θ`
    obj(θ, grd) # idem and stores the gradient in `grd`

"""
ObjectiveFunction{S}(H::AbstractMatrix{<:Real}, nsub::Integer) where {S} =
    ObjectiveFunction{S,float(eltype(H))}(H, nsub)

function ObjectiveFunction{S,T}(H::AbstractMatrix{<:Real},
                                nsub::Integer) where {S,T<:AbstractFloat}
    isnonnegative(H) || error(
        "entries of sources to currents matrix must be nonnegative")
    ncat, nsrc = size(H)
    n = nsrc + 1 # number of linear parameters
    return ObjectiveFunction{S,T}(
        H,
        Vector{T}(  undef, ncat), # c
        Vector{T}(  undef, ncat), # ∂c
        Vector{T}(  undef, nsub), # Δt
        Vector{Int}(undef, nsub), # l
        Vector{Int}(undef, nsub), # n
        Vector{T}(  undef, nsub), # avg
        Vector{T}(  undef, nsub), # var
        Vector{T}(  undef, ncat), # workspace w1
        Vector{T}(  undef, ncat), # workspace w2
        Vector{T}(  undef, ncat), # workspace w3
        Vector{T}(  undef, ncat), # workspace w4
        Matrix{T}(  undef, n, n), # LHS matrix A
        Vector{T}(  undef, n))    # RHS vector b
end

"""
    obj = ObjectiveFunction{S,T=eltype(cal)}(cal)

yields a workspace for fitting the parameters of the model of a detector pixel
in calibration data `cal`.  Type parameter `S` is the chosen parametrization
for the unknowns.  Type parameter `T` is the floating-point type for
computations.

"""
ObjectiveFunction{S}(cal::CalibrationData) where {S} =
    ObjectiveFunction{S,eltype(cal)}(cal)
ObjectiveFunction{S,T}(cal::CalibrationData) where {S,T<:AbstractFloat} =
    ObjectiveFunction{S,T}(cal.src_to_cat, length(cal.stat))

"""
    size(obj[, chk]) -> (nsub, ncat, nsrc)

yields numbers of subsets, of categories, and of sources in calibration data
`obj`.  By default, no cheks are performed but optional argument `chk`, can be
specified to request that the contents of `obj` be checked: if `chk` is
`:checksizes` or `Val(:checksizes)`, the sizes of internal buffers are checked;
if `chk` is `:checkindices` or `Val(:checkindices)`, the sizes of internal
buffers and the category indices are checked.

"""
function Base.size(obj::ObjectiveFunction)
    nsub = length(obj.l) # number of subsets
    ncat, nsrc = size(obj.H)
    return (nsub, ncat, nsrc)
end

@inline Base.size(obj::ObjectiveFunction, sym::Symbol) = size(obj, Val(sym))

function Base.size(obj::ObjectiveFunction, ::Val{:checksizes})
    nsub, ncat, nsrc = size(obj)
    n = nsrc + 1
    length(obj.c)   == ncat  || error("bad number of categories")
    length(obj.∂c)  == ncat  || error("bad number of category gradients")
    length(obj.Δt)  == nsub  || error("bad number of exposure times")
    length(obj.avg) == nsub  || error("bad number of empirical means")
    length(obj.var) == nsub  || error("bad number of empirical variances")
    length(obj.n)   == nsub  || error("bad number of subset sizes")
    length(obj.w1)  == ncat  || error("bad size for workspace W1")
    length(obj.w2)  == ncat  || error("bad size for workspace W2")
    length(obj.w3)  == ncat  || error("bad size for workspace W3")
    length(obj.w4)  == ncat  || error("bad size for workspace W4")
    size(obj.A)     == (n,n) || error("bad size for LHS matrix A")
    length(obj.b)   == n     || error("bad size for RHS vector b")
    return (nsub, ncat, nsrc)
end

function Base.size(obj::ObjectiveFunction, ::Val{:checkindices})
    nsub, ncat, nsrc = size(obj, :checksizes)
    flag = true
    @inbounds @simd for i in eachindex(obj.l)
        flag &= ((obj.l[i] - 1)%UInt < ncat)
    end
    flag || error("out of bound category index")
    return (nsub, ncat, nsrc)
end

"""
    extract!(obj, cal, k) -> obj

extracts into workspace `obj` the data for `k`-th pixel in calibration data
`cal`.  Argument `k` may be a tuple of Cartesian indices or an instance of
`CartesianIndex`.

"""
function extract!(obj::ObjectiveFunction,
                  cal::CalibrationData{T,N},
                  k::NTuple{N,Integer}) where {T,N}
    return extract!(obj, cal, CartesianIndex(k))
end
function extract!(obj::ObjectiveFunction,
                  cal::CalibrationData{T,N},
                  k::CartesianIndex{N}) where {T,N}
    return extract!(obj, cal,LinearIndices(size(cal.roi))[k])
end
function extract!(obj::ObjectiveFunction,
                  cal::CalibrationData{T,N},
                  k::Integer) where {T,N}
    return extract!(obj, cal, to_type(Int, k))
end
function extract!(obj::ObjectiveFunction,
                  cal::CalibrationData{T,N},
                  k::Int) where {T,N}
    nsub, ncat, nsrc = size(obj, :checksizes)
    length(cal.stat) == nsub || error(
        "fit workspace assumes a different number of subsets")
    @inbounds for (key, i) ∈ cal.stat_index
        # Extract (Δt,ℓ) for the subset of calibration data.
        cat       = key[1]             # category name
        obj.Δt[i] = key[2]             # exposure time
        obj.l[i]  = cal.cat_index[cat] # category index

        # Extract statistics of k-th pixel in subset of calibration data
        # samples.
        stat       = cal.stat[i]
        obj.n[i]   = nobs(stat)
        obj.avg[i] = mean(stat, k)
        obj.var[i] = var(stat, k; corrected=false)
    end
    return obj
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

lhs(eq::NormalEquations) = getfield(eq, :A)
rhs(eq::NormalEquations) = getfield(eq, :b)

Base.copy(eq::NormalEquations) = NormalEquations(copy(lhs(eq)), copy(rhs(eq)))

function NormalEquations(A::AbstractMatrix{<:Real}, b::AbstractVector{<:Real})
    T = float(promote_type(eltype(A), eltype(b)))
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
function (eq::NormalEquations)(x::AbstractVector{T}) where {T<:AbstractFloat}
    A, b = lhs(eq), rhs(eq)
    I = axes(b, 1)
    axes(A) == (I, I) || error(
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

function (eq::NormalEquations)(x::AbstractVector{T},
                               g::AbstractVector{T}) where {T<:AbstractFloat}
    A, b = lhs(eq), rhs(eq)
    I = axes(b, 1)
    axes(A) == (I, I) || error(
        "LHS matrix and RHS vector have incompatible indices")
    axes(x) == (I,) || error(
        "input variables have incompatible indices")
    axes(g) == (I,) || error(
        "output gradients have incompatible indices")
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
    fit_linear_terms!(obj, x; eta=Inf, kwds...)

fits the linear terms of the pixel detector model (the bias `z` and the source
terms).

Other keywords are passed to `vmlmb!`.

"""
function fit_linear_terms!(obj::ObjectiveFunction{S,T},
                           x::Vector{T} = zeros(T, size(obj.H,2) + 1);
                           reset::Bool = false,
                           nonnegative::Bool = true,
                           eta::Real = Inf,
                           mem::Integer = 5,
                           kwds...) where {S,T<:AbstractFloat}
    # Do a weighted least squares fit on all the linear parameters with the
    # positivity constraint on the source terms.
    eq = form_normal_equations!(obj, x, to_type(T, eta))
    reset && fill!(x, 0) # FIXME:
    if nonnegative
        # Solve the normal equations under the constraints that the source
        # terms are nonnegative.
        n = length(x)
        xmin = Vector{T}(undef, n) # FIXME: make is part of obj
        @inbounds for i in 1:n-1
            xmin[i] = 0 # source terms are nonnegative
        end
        xmin[n] = -Inf # z is unbounded
        vmlmb!(eq, x; mem=mem, lower=xmin, autodiff=false, kwds...)
    else
        x .= eq.A\eq.b # FIXME: use in-place operations
    end
    return x
end

"""
    form_normal_equations!(obj, x, η=Inf) -> eq

forms the normal equations for fitting the flux terms (the sources and the
zero-level).  Argument `η = g⋅σ² > 0` is the gain time the variance of the
read-out noise.  The weights are computed as:

    w[k,l] = n[k,l]/(c[l]⋅Δt[k,l] + η)

where `c = H*x[1:end-1]` is the flux per category of calibration with `H =
obj.H` the sources to categories matrix, `x[1:end-1]` the fluxes of the
sources, and `x[end]` the zero-level `z` (not used here).

If `η = Inf`, then the entries of `x` are not used at all and the weights are
assumed to be given by the number of samples: ` w[k,l] = n[k,l]`.

"""
function form_normal_equations!(obj::ObjectiveFunction{S,T},
                                x::AbstractVector{T},
                                η::Real = Inf) where {S,T<:AbstractFloat}
    return form_normal_equations!(obj, x, to_type(T, η))
end

function form_normal_equations!(obj::ObjectiveFunction{S,T},
                                x::AbstractVector{T},
                                η::T) where {S,T<:AbstractFloat}
    # Extract parameters from workspace.
    nsub, ncat, nsrc = size(obj, :checkindices)
    n = nsrc + 1
    length(x) == n || error("variables must have ", n, " elements")
    J = Base.OneTo(nsrc)
    L = Base.OneTo(ncat)
    H = obj.H

    # Determine whether flux dependent weights are to be computed.
    reweighted = false
    if η != Inf
        η > 0 || argument_error("value of `η = g⋅σ²` must be positive")
        @inbounds for j in J
            if x[j] != 0
                reweighted = true
                ((x[j] > 0) & isfinite(x[j])) || argument_error(
                    "sources `x[1:end-1]` must be finite and nonnegative")
            end
        end
    end

    # Integrate temporaries for all categories `l`:
    #
    #     w1[l] = sum_k w[k,l]*Δt[k,l]
    #     w2[l] = sum_k w[k,l]*Δt[k,l]^2
    #     w3[l] = sum_k w[k,l]*Δt[k,l]*d[k,l]
    #     Ann = A[n,n] = sum_{k,l} w[k,l]
    #     bn = b[n] = sum_{k,l} w[k,l]*d[k,l]
    #
    # with:
    #
    #    d[k,l] = obj.avg[k,l]
    #    w[k,l] = n[k,l]/(c[l]*Δt[k,l] + η)
    #
    # the "data" and the weights.  The sum is over index `i ~ (k,l)` such that
    # category index is given by `l = obj.l[i]`.
    #
    w1 = fill!(obj.w1, zero(T))
    w2 = fill!(obj.w2, zero(T))
    w3 = fill!(obj.w3, zero(T))
    Ann = zero(T)
    bn = zero(T)
    if reweighted
        # Compute fluxes in calibration categories and flux-dependent weights.
        c = mvmult!(obj.c, H, view(x, 1:nsrc))
        @inbounds for i in 1:nsub
            Δt  = obj.Δt[i]
            l   = obj.l[i]
            cΔt = c[l]*Δt
            w   = T(obj.n[i])/(cΔt + η)
            d   = obj.avg[i]
            wΔt = w*Δt
            w1[l] += wΔt
            w2[l] += wΔt*Δt
            w3[l] += wΔt*d
            Ann   += w
            bn    += w*d
        end
    else
        # Compute flux-independent weights.
        @inbounds for i in 1:nsub
            Δt  = obj.Δt[i]
            l   = obj.l[i]
            w   = T(obj.n[i])
            d   = obj.avg[i]
            wΔt = w*Δt
            w1[l] += wΔt
            w2[l] += wΔt*Δt
            w3[l] += wΔt*d
            Ann   += w
            bn    += w*d
        end
    end

    @inbounds begin
        # Compute the LHS matrix A of the normal equations.
        A = obj.A
        w4 = obj.w4 # FIXME: obj.c could be used as a temporary workspace here
        for j ∈ J
            # Leading (n-1)×(n-1) block.
            for l ∈ L
                w4[l] = H[l,j]*w2[l]
            end
            for jp ∈ J
                s = zero(T)
                for l ∈ L
                    s += w4[l]*H[l,jp]
                end
                A[j,jp] = s
                if jp == j
                    break
                end
                A[jp,j] = s
            end
            # Trailing column and row of A.
            let s = zero(T)
                for l ∈ L
                    s += H[l,j]*w1[l]
                end
                A[j,n] = s
                A[n,j] = s
            end
        end
        A[n,n] = Ann

        # Compute the RHS vector b of the normal equations.
        b = obj.b
        for j ∈ J
            s = zero(T)
            for l ∈ L
                s += H[l,j]*w3[l]
            end
            b[j] = s
        end
        b[n] = bn
    end # @inbounds

    # Return the normal equations.
    return NormalEquations{T}(A, b)
end

function max_readout_variance(obj::ObjectiveFunction{S,T}) where {S,T}
    nsub, ncat, nsrc = size(obj, :checkindices)
    N = 0 # to count total number of data
    s = zero(T) # to compute sum of data
    @inbounds @simd for i ∈ 1:nsub
        n  = obj.n[i]   # number of samples in subset
        d  = obj.avg[i] # data = sample mean
        s += n*d
        N += n
    end
    z = s/N
    s = zero(T) # to compute sum
    @inbounds @simd for i ∈ 1:nsub
        n   = obj.n[i]   # number of samples in subset
        d   = obj.avg[i] # data = sample mean
        v   = obj.var[i] # sample variance
        r   = z - d      # residuals: model - data
        s  += n*(v + r^2)
    end
    return s/N
end

function max_readout_variance(obj::ObjectiveFunction{S,T},
                              x::AbstractVector{T}) where {S,T}
    n = length(x)
    if n == 0
        return  max_readout_variance(obj)
    end
    nsub, ncat, nsrc = size(obj, :checkindices)
    if n == nsrc + 1
        # Get zero-kevel and compute fluxes in calibration categories.
        z = x[n]
        c = mvmult!(obj.c, obj.H, view(x, 1:n-1))
    elseif  n == nsrc + 3
        # Get zero-kevel and compute fluxes in calibration categories.
        z = x[1]
        c = mvmult!(obj.c, obj.H, view(x, 4:n))
    elseif n == 1
        z = x[1]
        c = fill!(obj.c, 0)
    else
        error("invalid length for vector of parameters")
    end

    N = 0 # to count total number of data
    s = zero(T) # to compute sum
    @inbounds @simd for i ∈ 1:nsub
        Δt  = obj.Δt[i]     # exposure time
        l   = obj.l[i]      # category index
        n   = obj.n[i]      # number of samples in subset
        cΔt = c[l]*Δt       # contribution of sources
        d   = obj.avg[i]    # data = sample mean
        v   = obj.var[i]    # sample variance
        r   = (cΔt + z) - d # residuals: model - data
        s  += n*(v + r^2)
        N  += n
    end
    return s/N
end

#
#     f = obj(z, g, η, s)
#
# yields the value of the objective function associated with workspace `obj`
# and for model parameters `x = (z, g, η, s...)` with `z` the zero level, `g`
# the gain, `η` the variance of the readout noise times the gain and the source
# terms `s`.
#
function (obj::ObjectiveFunction{:alt,T})(z::Real, g::Real, η::Real,
                                          s::AbstractVector{T}) where {T}
    return obj(to_type(T, z), to_type(T, g), to_type(T, η), s)
end

function (obj::ObjectiveFunction{:alt,T})(z::T, g::T, η::T,
                                          s::AbstractVector{T}) where {T}
    check_args(obj, z, g, η, s)
    c = mvmult!(obj.c, obj.H, s) # compute fluxes in calibration categories
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    nsub, ncat, nsrc = size(obj, :checkindices)
    @inbounds @simd for i ∈ 1:nsub
        Δt  = obj.Δt[i]     # exposure time
        l   = obj.l[i]      # category index
        n   = obj.n[i]      # number of samples in subset
        n_  = T(n)
        cΔt = c[l]*Δt       # contribution of sources
        d   = obj.avg[i]    # data = sample mean
        r   = (cΔt + z) - d # residuals: model - data
        w   = n_/(cΔt + η)  # weight
        χ² += w*(obj.var[i] + r^2)
        sum_n_logw_n += n_*log(w/n_)
        N += n
    end
    return g*χ² - sum_n_logw_n - N*log(g)
end

function check_args(obj::ObjectiveFunction{:alt},
                    z::Real, g::Real, η::Real,
                    s::AbstractVector{<:Real})
    g > 0 || throw_argument_error("gain `g` must be positive")
    η > 0 || throw_argument_error("readout variance `η` must be positive")
    isnonnegative(s) || throw_argument_error(
        "source terms `s` must be nonnegative")
end

#
#     f = obj(z, g, η, s, grd)
#
# yields the value of the objective function `f(x)` associated with workspace
# `obj` and overwrites `grd` with the gradient `∇f(x)` for model parameters `x
# = (z, g, η, s...)` with `z` the zero level, `g` the gain, `η` the variance of
# the readout noise times the gain and the source terms `s`.
#
function (obj::ObjectiveFunction{:alt,T})(z::Real, g::Real, η::Real,
                                          s::AbstractVector{T},
                                          grd::AbstractVector{T}) where {T}
    return obj(to_type(T, z), to_type(T, g), to_type(T, η), s, grd)
end

function (obj::ObjectiveFunction{:alt,T})(z::T, g::T, η::T,
                                          s::AbstractVector{T},
                                          grd::AbstractVector{T}) where {T}
    check_args(obj, z, g, η, s)
    length(grd) == 3 + length(s) || error("bad gradient size")
    c = mvmult!(obj.c, obj.H, s) # compute fluxes in calibration categories
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    ∂c = fill!(obj.∂c, 0) # to compute ∂L/∂c
    ∂z = zero(T) # to compute ∂L/∂z
    ∂η = zero(T) # to compute ∂L/∂η
    nsub, ncat, nsrc = size(obj, :checkindices)
    @inbounds @simd for i ∈ 1:nsub
        Δt  = obj.Δt[i]     # exposure time
        l   = obj.l[i]      # category index
        n   = T(obj.n[i])   # number of samples in subset
        cΔt = c[l]*Δt       # contribution of sources
        d   = obj.avg[i]    # data = sample mean
        r   = (cΔt + z) - d # residuals: model - data
        q   = cΔt + η       # model variance times gain
        w   = n/q           # weight
        v   = obj.var[i] + r^2
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
    @inbounds grd[1] = ∂z
    @inbounds grd[2] = ∂g
    @inbounds grd[3] = ∂η
    mvmult!(view(grd, 4:length(grd)), obj.H', ∂c)
    # Return total objective function.
    return g*χ² - sum_n_logw_n - N*log(g)
end

#     f = obj(x, grd)
#
# yields the value of the objective function `f(x)` associated with workspace
# `obj` and overwrites `grd` with the gradient `∇f(x)` for model parameters
# `x = (z, g, σ, s...)` with `z` the zero level, `g` the gain, `σ` the standard
# deviation of the readout noise and the source terms `s`:
#
#     x[1]     = z # bias or zero-level (ADU)
#     x[2]     = g # gain (e-/ADU)
#     x[3]     = σ # standard deviation of read-out noise (ADU)
#     x[4:end] = s # source terms (ADU/s)
#
# Usage example:
#
#     xmin = zeros(T, length(x))
#     xmin[1] = -Inf       # min. for z
#     xmin[2] = 1.0        # min. for g
#     xmin[3] = sqrt(1/12) # min. for σ
#     vmlmb(obj, copy(x); lower=xmin, verb=1, mem=length(x), maxiter=1000,
#           maxeval=5000, ftol=(0,0), xtol=(0,0), gtol=(1e-5,0))
#
function (obj::ObjectiveFunction{:orig,T})(x::Vector{T},
                                           grd::Vector{T}) where {T}
    nsub, ncat, nsrc = size(obj, :checkindices)
    inds = axes(x)
    axes(grd) == inds || error(
        "variables and gradients have different indices")
    first(inds[1]) == 1 || error(
        "variables have non-standard indexing")
    xlen = length(x)
    xlen == nsrc + 3 ||  error(
        "variables must have ", nsrc + 3, " elements, got ", xlen)
    @inbounds begin
        # Unpack parameters.
        z = x[1]
        g = x[2]
        σ = x[3]
        I = 4:xlen # index range of source terms
        (isfinite(g) && g > 0) || argument_error(
            "gain must be finite and strictly positive")
        (isfinite(σ) && σ > 0) || argument_error(
            "standard deviation of read-out ",
            "noise must be finite and strictly positive")
        ρ = 1/g
        σ² = σ

        # Compute fluxes in calibration categories.
        c = mvmult!(obj.c, obj.H, view(x, I))

        # Loop to integrate objective function and its gradient.
        f      = zero(T)          # to compute f
        ∂f_∂z  = zero(T)          # to compute ∂f/∂z
        ∂f_∂ρ  = zero(T)          # to compute ∂f/∂ρ
        ∂f_∂σ² = zero(T)          # to compute ∂f/∂σ²
        ∂f_∂c  = fill!(obj.∂c, 0) # to compute ∂f/∂c
        @simd for i ∈ 1:nsub
            # Objective function for this subset.
            Δt  = obj.Δt[i]        # exposure time
            l   = obj.l[i]         # category index
            n   = T(obj.n[i])      # number of samples in subset
            cΔt = c[l]*Δt          # contribution of sources
            m   = cΔt + z          # model of data
            r   = m - obj.avg[i]   # residuals = model - sample mean
            q   = r^2 + obj.var[i] # quadratic error
            v   = ρ*cΔt + σ²       # model of variance
            w   = 1/v              # weight
            f  += n*(w*q - log(w)) # objective function

            # Partial derivatives of `f` w.r.t. `m` and `w`:
            ∂f_∂w = n*(q - v)   # ∂f/∂w = n⋅(q - 1/w)
            ∂f_∂m = 2n*w*r      # ∂f/∂m = 2n⋅w⋅r

            # ∂(m,w)/∂z = (1, 0)
            ∂f_∂z += ∂f_∂m

            # ∂(m,w)/∂ρ = (0, -w²⋅c⋅Δt)
            w² = w^2
            ∂f_∂ρ -= w²*cΔt*∂f_∂w

            # ∂(m,w)/∂σ² = (0, -w²)
            ∂f_∂σ² -= w²*∂f_∂w

            # ∂(m,w)/∂c = (1, -w²⋅ρ)⋅Δt
            ∂f_∂c[l] += (∂f_∂m - w²*ρ*∂f_∂w)*Δt
        end

        # Store/convert gradients and return objective function.
        ∂f_∂g = -ρ^2*∂f_∂ρ
        ∂f_∂σ = 2σ*∂f_∂σ²
        grd[1] = ∂f_∂z
        grd[2] = ∂f_∂g
        grd[3] = ∂f_∂σ
        mvmult!(view(grd, I), obj.H', ∂f_∂c)
        return f
    end # @inbounds
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

end # module
