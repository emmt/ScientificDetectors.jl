module Calibration

export
    cologlikelihood,
    detectorbias,
    detectorgain,
    detectornoise,
    validpixelmap,
    sources,
    sourcesid,
    nsources

using ProgressMeter, Distributions
using StatsBase, Statistics, LinearAlgebra
using SimpleExpressions
using AsType, ArrayTools, StructuredArrays
using OptimPackNextGen
using MultivariateOnlineStatistics
using MultivariateOnlineStatistics:
    storage
using StructuredArrays
using ..ScientificDetectors
using ..ScientificDetectors:
    DetectorAxisTypes,
    Identifiers,
    OnlineStatistics,
    binning,
    getfields,
    identifier,
    nth,
    offset
import ..ScientificDetectors:
    DetectorAxes,
    argument_error,
    dimension_mismatch,
    exposuretime
import Base: push!, merge!

const Category = Union{AbstractString,Symbol}

const Colons{N} = NTuple{N,Colon}

# Include code for types with constructors and basic method.
include("ReducedCalibration.jl")
include("CalibrationDataFrame.jl")
include("CalibrationData.jl")
include("CalibrationFrameSampler.jl")
include("SimpleCalibration.jl")
include("badpixel.jl")

#------------------------------------------------------------------------------
# FIXME: Only needed by ReducedCalibration.
"""
    find(obj, key) -> j

yields the index `j` of the source term in reduced calibration data which
match `key` or `0` if not found.

"""
find(obj::ReducedCalibration, key::Nothing) = 0

function find(obj::ReducedCalibration, key::AbstractString)
    src = sourcesid(obj)
    n = 0
    j = 0
    for i in 1:nsources(obj)
        if src[i] == key
            j = i
            n += 1
        end
    end
    n > 1 && error("non-unique source identifier")
    return j
end

find(obj::ReducedCalibration, j::Integer) =
    (1 ≤ j ≤  nsources(obj) ? Int(j) : 0)

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
    # Buffers needed for the data and the model (only modified by the
    # `extract!` method).
    H::Matrix{T}   # sources to currents matrix
    Δt::Vector{T}  # exposure times
    l::Vector{Int} # indices of categories
    n::Vector{Int} # number of samples in subset
    avg::Vector{T} # empirical subset mean
    var::Vector{T} # empirical (biased) subset variance

    # Workspaces of length `ncat`.
    wcat1::Vector{T} # used to compute `c` and `∂f/∂c`
    wcat2::Vector{T}
    wcat3::Vector{T}
    wcat4::Vector{T}

    # Workspaces of length `nsub`.
    wsub1::Vector{T} # used to compute `c⋅Δt`

    # Workspaces for the normal equations.
    sz::Vector{T} # to store [s..., z]
    sz_min::Vector{T} # to store inferoir bound for [s..., z]
    A::Matrix{T} # LHS matrix of the normal equations
    b::Vector{T} # RHS vector of the normal equations
end

"""
    obj = ObjectiveFunction{S,T=float(eltype(H))}(H, nsub)

yields a workspace for fitting the parameters of the model of a detector pixel
for sources to currents matrix `H` and `nsub` data subsets.  All entries of `H`
must be nonnegative (this is checked).  The same workspace can be re-used for
another pixel provided the matrix `H` remains the same.  Type parameter `S` is
a symbol which specifies the chosen parametrization for the unknowns `θ`:

- `S = :zgσs` to have parameters `θ = [z, g, σ, s...]`;

- `S = :zgηs` to have parameters `θ = [z, g, η, s...]`;

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
        Vector{T}(  undef, nsub), # Δt
        Vector{Int}(undef, nsub), # l
        Vector{Int}(undef, nsub), # n
        Vector{T}(  undef, nsub), # avg
        Vector{T}(  undef, nsub), # var
        Vector{T}(  undef, ncat), # workspace wcat1
        Vector{T}(  undef, ncat), # workspace wcat2
        Vector{T}(  undef, ncat), # workspace wcat3
        Vector{T}(  undef, ncat), # workspace wcat4
        Vector{T}(  undef, nsub), # workspace wsub1
        Vector{T}(  undef, n),    # vector sz
        Vector{T}(  undef, n),    # vector sz_min
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

# Conversion constructors.  The same object is returned if possible, call
# `copy(obj)` to make a distinctive copy.
ObjectiveFunction(obj::ObjectiveFunction) = obj
ObjectiveFunction{S}(obj::ObjectiveFunction{S}) where {S} = obj
ObjectiveFunction{S}(obj::ObjectiveFunction{<:Any,T}) where {S,T} =
    ObjectiveFunction{S,T}(obj)
ObjectiveFunction{S,T}(obj::ObjectiveFunction{S,T}) where {S,T} = obj
function ObjectiveFunction{S,T}(obj::ObjectiveFunction{<:Any,T}) where {S,T}
    isa(S, Symbol) || argument_error("type parameter S must be a symbol")
    return ObjectiveFunction{S,T}(getfields(obj)...)
end

Base.convert(::Type{T}, obj::ObjectiveFunction) where {T<:ObjectiveFunction} =
    T(obj)

# Copy constructor.
Base.copy(obj::ObjectiveFunction{S,T}) where {S,T} =
    ObjectiveFunction{S,T}(map(copy, getfields(obj))...)

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
    nsub = length(obj.Δt) # number of subsets
    ncat, nsrc = size(obj.H)
    return (nsub, ncat, nsrc)
end

@inline Base.size(obj::ObjectiveFunction, sym::Symbol) = size(obj, Val(sym))

function Base.size(obj::ObjectiveFunction, ::Val{:checksizes})
    nsub, ncat, nsrc = size(obj)
    n = nsrc + 1
    #=
    size(obj.H) == (ncat, nsrc) || error(
        "bad size of sources to categories matrix")
    length(obj.Δt)  == nsub  || error("bad number of exposure times")
    =#
    length(obj.l)      == nsub  || error("bad number of category indices")
    length(obj.n)      == nsub  || error("bad number of subset sizes")
    length(obj.avg)    == nsub  || error("bad number of empirical means")
    length(obj.var)    == nsub  || error("bad number of empirical variances")
    length(obj.wcat1)  == ncat  || error("bad size for workspace WCAT1")
    length(obj.wcat2)  == ncat  || error("bad size for workspace WCAT2")
    length(obj.wcat3)  == ncat  || error("bad size for workspace WCAT3")
    length(obj.wcat4)  == ncat  || error("bad size for workspace WCAT4")
    length(obj.wsub1)  == nsub  || error("bad size for workspace WSUB1")
    length(obj.sz)     == n     || error("bad size for field `sz`")
    length(obj.sz_min) == n     || error("bad size for field `sz_min`")
    size(obj.A)        == (n,n) || error("bad size for LHS matrix A")
    length(obj.b)      == n     || error("bad size for RHS vector b")
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
    return extract!(obj, cal, as(Int, k))
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

"""
    reset!(obj) -> obj

reset the workspace by setting all array to zero.

"""
function reset!(obj::ObjectiveFunction)
    # Workspaces of length `ncat`.
    fill!(obj.wcat1,0)
    fill!(obj.wcat2,0)
    fill!(obj.wcat3,0)
    fill!(obj.wcat4,0)

    # Workspaces of length `nsub`.
    fill!(obj.wsub1,0)

    # Workspaces for the normal equations.
    fill!(obj.sz,0)
    fill!(obj.sz_min,0)
    fill!(obj.A,0)
    fill!(obj.b,0)
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
                           x::AbstractVector{T} = zeros(T, size(obj.H,2) + 3);
                           reset::Bool = false,
                           nonnegative::Bool = true,
                           eta::Real = Inf,
                           mem::Integer = 5,
                           kwds...) where {S,T<:AbstractFloat}
    # Do a weighted least squares fit on all the linear parameters with the
    # positivity constraint on the source terms.
    eq = form_normal_equations!(obj, x, as(T, eta))
    if nonnegative
        # Solve the normal equations under the constraints that the source
        # terms are nonnegative.
        fill!(obj.sz, 0)
        fill!(obj.sz_min, 0)
        obj.sz_min[end] = -Inf
        vmlmb!(eq, obj.sz; mem=mem, lower=obj.sz_min, autodiff=false, kwds...)
    else
        obj.sz .= eq.A\eq.b # FIXME: use in-place operations
    end
    x[1] = obj.sz[end]
    x[4:end] = obj.sz[1:end-1]
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
    return form_normal_equations!(obj, x, as(T, η))
end

function form_normal_equations!(obj::ObjectiveFunction{S,T},
                                x::AbstractVector{T},
                                η::T) where {S,T<:AbstractFloat}
    # Extract parameters from workspace.
    nsub, ncat, nsrc = size(obj, :checkindices)
    length(x) == nsrc + 3 || error("variables must have ", nsrc + 3, " elements")
    s = view(x, 4:length(x))
    n = nsrc + 1 # bias z is the last source
    J = Base.OneTo(nsrc)   # only index over the sources
    L = Base.OneTo(ncat)
    H = obj.H

    # Determine whether flux dependent weights are to be computed.
    reweighted = false
    if η != Inf
        η > 0 || argument_error("value of `η = g⋅σ²` must be positive")
        @inbounds for j in eachindex(s)
            s[j] == 0 && continue
            reweighted = true
            ((s[j] > 0) & isfinite(x[j])) || argument_error(
                "source terms `s = x[4:end]` must be finite and nonnegative")
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
    w1 = fill!(obj.wcat1, zero(T))
    w2 = fill!(obj.wcat2, zero(T))
    w3 = fill!(obj.wcat3, zero(T))
    Ann = zero(T)
    bn = zero(T)
    if reweighted
        # Compute fluxes in calibration categories and flux-dependent weights.
        c = compute_c(obj, s)
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
        w4 = obj.wcat4
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

function compute_σ²_max(obj::ObjectiveFunction{S,T}) where {S,T}
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

function compute_σ²_max(obj::ObjectiveFunction{S,T},
                        x::AbstractVector{T}) where {S,T}
    n = length(x)
    if n == 0
        return  compute_σ²_max(obj)
    end
    nsub, ncat, nsrc = size(obj, :checkindices)
    if n == nsrc + 1
        # Get zero-kevel and compute fluxes in calibration categories.
        z = x[n]
        c = compute_c(obj, view(x, 1:n-1))
    elseif  n == nsrc + 3
        # Get zero-kevel and compute fluxes in calibration categories.
        z = x[1]
        c = compute_c(obj, view(x, 4:n))
    elseif n == 1
        z = x[1]
        c = fill!(obj.wcat1, 0)
    else
        error("invalid length for vector of parameters")
    end

    # Compute contribution of sources.
    cΔt = compute_cΔt(obj, c; unsafe=true)

    N = 0 # to count total number of data
    s = zero(T) # to compute sum
    @inbounds @simd for i ∈ 1:nsub
        n   = obj.n[i]      # number of samples in subset
        d   = obj.avg[i]    # data = sample mean
        v   = obj.var[i]    # sample variance
        r   = (cΔt[i] + z) - d # residuals: model - data
        s  += n*(v + r^2)
        N  += n
    end
    return s/N
end

function unpack_parameters(obj::ObjectiveFunction{<:Any,T},
                           x::AbstractVector{T},
                           grd::AbstractVector{T};
                           nonnegative::Bool = false) where {T}
    axes(grd) == axes(x) || error(
        "variables and gradients have different indices")
    return unpack_parameters(obj, x; nonnegative=nonnegative)
end

function unpack_parameters(obj::ObjectiveFunction{:zgσs,T},
                           x::AbstractVector{T};
                           nonnegative::Bool = false) where {T}
    nsub, ncat, nsrc = size(obj, :checkindices)
    Base.has_offset_axes(x) && error(
        "variables have non-standard indexing")
    n = length(x)
    n == nsrc + 3 || error(
        "variables must have ", nsrc + 3, " elements, got ", n)
    @inbounds z = x[1]
    @inbounds g = x[2]
    (isfinite(g) && g > 0) || argument_error(
        "gain `g` must be finite and strictly positive")
    @inbounds σ = x[3]
    (isfinite(σ) && σ > 0) || argument_error(
        "standard deviation of the read-out noise `σ` ",
        "must be finite and strictly positive")
    s = view(x, 4:n) # source terms
    if nonnegative
        isnonnegative(s) || argument_error("source terms must be nonnegative")
    end
    return z, g, σ, s
end

function unpack_parameters(obj::ObjectiveFunction{:zgηs,T},
                           x::AbstractVector{T};
                           nonnegative::Bool = false) where {T}
    nsub, ncat, nsrc = size(obj, :checkindices)
    Base.has_offset_axes(x) && error(
        "variables have non-standard indexing")
    n = length(x)
    n == nsrc + 3 || error(
        "variables must have ", nsrc + 3, " elements, got ", n)
    @inbounds z = x[1]
    @inbounds g = x[2]
    (isfinite(g) && g > 0) || argument_error(
        "gain `g` must be finite and strictly positive")
    @inbounds η = x[3]
    (isfinite(η) && η > 0) || argument_error(
        "parameter `η` (the variance of the read-out noise times the gain) ",
        "must be finite and strictly positive")
    s = view(x, 4:n) # source terms
    if nonnegative
        isnonnegative(s) || argument_error("source terms must be nonnegative")
    end
    return z, g, η, s
end

function unpack_parameters(obj::ObjectiveFunction{:zρσs,T},
                           x::AbstractVector{T};
                           nonnegative::Bool = false) where {T}
    nsub, ncat, nsrc = size(obj, :checkindices)
    Base.has_offset_axes(x) && error(
        "variables have non-standard indexing")
    n = length(x)
    n == nsrc + 3 || error(
        "variables must have ", nsrc + 3, " elements, got ", n)
    @inbounds z = x[1]
    @inbounds ρ = x[2]
    (isfinite(g) && g > 0) || argument_error(
        "reciprocal gain `ρ` must be finite and strictly positive")
    @inbounds σ = x[3]
    (isfinite(σ) && σ > 0) || argument_error(
        "standard deviation of the read-out noise `σ` ",
        "must be finite and strictly positive")
    s = view(x, 4:n) # source terms
    if nonnegative
        isnonnegative(s) || argument_error("source terms must be nonnegative")
    end
    return z, ρ, σ, s
end

function unpack_parameters_with_cΔt(obj::ObjectiveFunction{<:Any,T},
                                    x::AbstractVector{T},
                                    grd::AbstractVector{T};
                                    nonnegative::Bool = false) where {T}
    z, g, q, s = unpack_parameters(obj, x, grd; nonnegative=nonnegative)
    cΔt = compute_cΔt(obj, compute_c(obj, s); unsafe=true)
    return z, g, q, cΔt
end

function unpack_parameters_with_cΔt(obj::ObjectiveFunction{<:Any,T},
                                    x::AbstractVector{T};
                                    nonnegative::Bool = false) where {T}
    z, g, q, s = unpack_parameters(obj, x; nonnegative=nonnegative)
    cΔt = compute_cΔt(obj, compute_c(obj, s); unsafe=true)
    return z, g, q, cΔt
end

#
#     f = obj(x)
#
# yields the value of the objective function associated with workspace `obj`
# and for model parameters `x = (z, g, η, s...)` with `z` the zero level, `g`
# the gain, `η` the variance of the readout noise times the gain and the source
# terms `s`.
#
#
#     f = obj(x, grd)
#
# yields the value of the objective function `f(x)` associated with workspace
# `obj` and overwrites `grd` with the gradient `∇f(x)` for model parameters `x
# = (z, g, η, s...)` with `z` the zero level, `g` the gain, `η` the variance of
# the readout noise times the gain and the source terms `s`.
#

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
#
# Parameters: x = (z, g, σ, s...)
#
function (obj::ObjectiveFunction{:zgσs,T})(x::AbstractVector{T}) where {T}
    # Unpack parameters and check arguments.
    z, g, σ, cΔt = unpack_parameters_with_cΔt(obj, x)
    ρ = 1/g
    σ² = σ^2
    nsub = length(cΔt)

    # Loop to integrate the objective function.
    f = zero(T) # to compute f
    @inbounds @simd for i ∈ 1:nsub
        n  = T(obj.n[i])          # number of samples in subset
        u  = cΔt[i]               # contribution of sources
        r  = (u + z) - obj.avg[i] # residuals = model - sample mean
        u₊ = fastmax(u, zero(T))
        v  = ρ*u₊ + σ²            # model of variance
        χ² = (r^2 + obj.var[i])/v # χ² per sample of the sub-set
        f += (log(v) + χ²)*n      # objective function
    end
    return f
end

function (obj::ObjectiveFunction{:zgσs,T})(x::AbstractVector{T},
                                           grd::AbstractVector{T}) where {T}
    # Unpack parameters and check arguments.
    z, g, σ, cΔt = unpack_parameters_with_cΔt(obj, x, grd)
    ρ = 1/g
    σ² = σ^2
    nsub = length(cΔt)

    # Loop to integrate objective function and its gradient.
    f      = zero(T) # to compute f
    ∂f_∂z  = zero(T) # to compute ∂f/∂z
    ∂f_∂ρ  = zero(T) # to compute ∂f/∂ρ
    ∂f_∂σ² = zero(T) # to compute ∂f/∂σ²
    ∂f_∂c_temp = cΔt # overwrite workspace cΔt for ∂f/∂c
    @inbounds @simd for i ∈ 1:nsub
        Δt = obj.Δt[i]            # exposure time
        n  = T(obj.n[i])          # number of samples in subset
        u  = cΔt[i]               # contribution of sources
        r  = (u + z) - obj.avg[i] # residuals = model - sample mean
        u₊ = fastmax(u, zero(T))
        v  = ρ*u₊ + σ²            # model of variance
        w  = 1/v                  # weight
        χ² = (r^2 + obj.var[i])*w # χ² per sample of the sub-set
        f += (log(v) + χ²)*n      # objective function
        nw = n*w
        ∂f_∂m = 2*nw*r            # ∂f/∂m[k,l]
        ∂f_∂v = nw*(1 - χ²)       # ∂f/∂v[k,l]
        ∂f_∂z  += ∂f_∂m
        ∂f_∂σ² += ∂f_∂v
        ∂f_∂ρ  += u₊*∂f_∂v
        ∂v_∂u = ifelse(u > zero(T), ρ, zero(T))
        ∂f_∂c_temp[i] = (∂f_∂m + ∂f_∂v*∂v_∂u)*Δt
    end

    # Convert gradients and return objective function.
    ∂f_∂g = -ρ^2*∂f_∂ρ # ∂f/∂g = -ρ²⋅(∂f/∂ρ)
    ∂f_∂σ = 2σ*∂f_∂σ²  # ∂f/∂σ = 2⋅σ⋅(∂f/∂σ²)
    @inbounds grd[1] = ∂f_∂z
    @inbounds grd[2] = ∂f_∂g
    @inbounds grd[3] = ∂f_∂σ
    ∂f_∂c = fill!(obj.wcat1, zero(T)) # to compute ∂f/∂c
    @inbounds @simd for i ∈ 1:nsub
        l = obj.l[i] # category index
        ∂f_∂c[l] += ∂f_∂c_temp[i]
    end
    mvmult!(view(grd, 4:length(grd)), obj.H', ∂f_∂c)
    return f
end

#
# Parameters: x = (z, g, η, s...)
#
function (obj::ObjectiveFunction{:zgηs,T})(x::AbstractVector{T}) where {T}
    # Unpack parameters and check arguments.
    z, g, η, cΔt = unpack_parameters_with_cΔt(obj, x)
    ρ = 1/g
    nsub = length(cΔt)

    # Loop to integrate the objective function.
    f = zero(T) # to compute f
    @inbounds @simd for i ∈ 1:nsub
        n  = T(obj.n[i])          # number of samples in subset
        u  = cΔt[i]               # contribution of sources
        r  = (u + z) - obj.avg[i] # residuals = model - sample mean
        u₊ = fastmax(u, zero(T))
        v  = (u₊ + η)*ρ           # model of variance
        χ² = (r^2 + obj.var[i])/v # χ² per sample of the sub-set
        f += (log(v) + χ²)*n      # objective function
    end
    return f
end

function (obj::ObjectiveFunction{:zgηs,T})(x::AbstractVector{T},
                                           grd::AbstractVector{T}) where {T}
    # Unpack parameters and check arguments.
    z, g, η, cΔt = unpack_parameters_with_cΔt(obj, x, grd)
    ρ = 1/g
    nsub = length(cΔt)

    # Loop to integrate the objective function and its gradient.
    f          = zero(T) # to compute f
    ∂f_∂z      = zero(T) # to compute ∂f/∂z = sum_{k,l} ∂f/∂m[k,l]
    ∂f_∂σ²     = zero(T) # to compute ∂f/∂σ² = sum_{k,l} ∂f/∂v[k,l]
    sum_v∂f_∂v = zero(T) # to compute sum_{k,l} v[k,l]⋅(∂f/∂v[k,l])
    ∂f_∂c_temp = cΔt     # overwrite workspace cΔt for ∂f/∂c
    @inbounds @simd for i ∈ 1:nsub
        Δt = obj.Δt[i]            # exposure time
        n  = T(obj.n[i])          # number of samples in subset
        u  = cΔt[i]               # contribution of sources
        r  = (u + z) - obj.avg[i] # residuals = model - sample mean
        u₊ = fastmax(u, zero(T))
        v  = (u₊ + η)*ρ           # model of variance
        w  = 1/v                  # weight
        χ² = (r^2 + obj.var[i])*w # χ² per sample of the sub-set
        f += (log(v) + χ²)*n      # objective function
        nw = n*w
        ∂f_∂m = 2*nw*r            # ∂f/∂m[k,l]
        ∂f_∂v = nw*(1 - χ²)       # ∂f/∂v[k,l]
        ∂f_∂z += ∂f_∂m
        ∂f_∂σ² += ∂f_∂v
        sum_v∂f_∂v += v*∂f_∂v
        ∂v_∂u = ifelse(u > zero(T), ρ, zero(T))
        ∂f_∂c_temp[i] = (∂f_∂m + ∂f_∂v*∂v_∂u)*Δt
    end

    # Convert gradients and return objective function.
    ∂f_∂g = -ρ*sum_v∂f_∂v # ∂f/∂g = -(1/g)⋅sum_{k,l} v[k,l]⋅(∂f/∂v[k,l])
    ∂f_∂η = ρ*∂f_∂σ²      # ∂f/∂η = (1/g)⋅(∂f/∂σ²)
    @inbounds grd[1] = ∂f_∂z
    @inbounds grd[2] = ∂f_∂g
    @inbounds grd[3] = ∂f_∂η
    ∂f_∂c = fill!(obj.wcat1, zero(T)) # to compute ∂f/∂c
    @inbounds @simd for i ∈ 1:nsub
        l = obj.l[i] # category index
        ∂f_∂c[l] += ∂f_∂c_temp[i]
    end
    mvmult!(view(grd, 4:length(grd)), obj.H', ∂f_∂c)
    return f
end

"""
    compute_c(obj, s) -> c

yields the current terms `c` corresponding to the source terms `s` for the
calibration data stored by `obj`.  The returned array is the internal buffer
`obj.wcat1` of `obj`.

"""
function compute_c(obj::ObjectiveFunction{S,T},
                   s::AbstractVector{T}) where {S,T}
    return mvmult!(obj.wcat1, obj.H, s)
end

"""
    compute_cΔt(obj [, c = obj.wcat1]; unsafe=false) -> cΔt

yields the contribution of the sources `c` in the calibration data stored by
`obj` using the internal buffer `obj.wsub1` and returns it.  If `c` is not
specified, the internal buffer of `obj` storing the sources is used (this
assumes that its contents has been updated).  Computations are done by
`compute_cΔt!`.

"""
function compute_cΔt(obj::ObjectiveFunction{S,T},
                     c::AbstractVector{T} = obj.wcat1;
                     kwds...) where {S,T}
    return compute_cΔt!(obj.wsub1, obj, c; kwds...)
end

"""
    compute_cΔt!(cΔt, obj, c; unsafe=false) -> cΔt

overwrites destination array `cΔt` with the contribution of the sources `c` in
the calibration data stored by `obj` and returns `cΔt`.  This amounts to
computing `cΔt[i] = c[obj.l[i]]*obj.Δt[i]` for all indices `i` and returns
`cΔt`.  This method also checks that arguments and internal buffers of `obj`
have correct indices.

If keyword `unsafe` is true, it is assumed that the category indices in `obj`
can be trusted.

"""
function compute_cΔt!(cΔt::AbstractVector{T},
                      obj::ObjectiveFunction{S,T},
                      c::AbstractVector{T};
                      unsafe::Bool=false) where {S,T}
    nsub, ncat, nsrc = size(obj)
    I = Base.OneTo(nsub)
    L = Base.OneTo(ncat)
    axes(cΔt) == (I,) || argument_error(
        "destination array has incompatible indices")
    axes(obj.Δt) == (I,) || argument_error(
        "exposure time buffer has incompatible indices")
    axes(obj.l) == (I,) || argument_error(
        "category index buffer has incompatible indices")
    axes(c) == (L,) || argument_error(
        "category array has incompatible indices")
    if !unsafe
        flag = true
        @inbounds @simd for i ∈ I
            flag &= ((obj.l[i] - 1)%UInt < ncat)
        end
        flag || error("out of bound category index")
    end
    @inbounds for i ∈ I
        Δt     = obj.Δt[i] # exposure time
        l      = obj.l[i]  # category index
        cΔt[i] = c[l]*Δt   # contribution of sources
    end
    return cΔt
end

ReducedCalibration(dat::CalibrationData; kwds...) =
    ReducedCalibration(:zgσs, dat; kwds...)

ReducedCalibration(alg::Symbol, dat::CalibrationData; kwds...) =
    ReducedCalibration(Val(alg), dat; kwds...)

function ReducedCalibration(alg::Val{S},
                            dat::CalibrationData{T,N};
                            vpm::AbstractArray{Bool, N} = FastUniformArray(true, size(dat)),
                            nonnegative::Bool = true,
                            maxval::Real = +Inf,
                            gmin::Real = 0.1,
                            gmax::Real = +Inf,
                            g::Real = gmin,
                            σ::Real = 1/sqrt(12),
                            badpixvalue::T = T(0),
                            quiet::Bool = false) where {S,T,N}
    axes(vpm) == axes(dat) || throw(DimensionMismatch("incompatible indices"))
    (isfinite(gmin) && gmin > 0) || argument_error(
        "value of keyword `gmin` must be finite and positive")
    (isfinite(g) && g ≥ gmin) || argument_error(
        "value of keyword `g` must be finite and greater or equal `gmin`")
    (isfinite(σ) && σ > 0) || argument_error(
        "value of keyword `σ` must be finite and positive")
    nthreads = Threads.nthreads()
    if nthreads ≤ 1
        @warn "You may start Julia as `JULIA_NUM_THREADS=$(Base.Sys.CPU_THREADS) julia`"
    end

    obj = [ObjectiveFunction{S}(dat) for i in 1:nthreads]
    nsub, ncat, nsrc = size(obj[1])
    n = 3 + nsrc
    x = [Vector{T}(undef, n) for i in 1:nthreads]
    xmin = [Vector{T}(undef, n) for i in 1:nthreads]
    xmax = [Vector{T}(undef, n) for i in 1:nthreads]
    for i in 1:nthreads
        fill!(xmin[i], -Inf)
        fill!(xmax[i], +Inf)
        xmin[i][2] = gmin
        xmax[i][2] = gmax
        xmin[i][3] = 1e-6
        fill!(view(xmax[i], 4:n), maxval)
        if nonnegative
            fill!(view(xmin[i], 4:n), 0)
        end
    end

    inits(::Type{T}, dims::Dims{N}, value::T) where {T<:AbstractFloat,N} =
        fill!(Array{T,N}(undef, dims), value)
    dims = size(dat.roi)
    src_names = Array{String}(undef, nsrc)
    for (key,val) in dat.src_index
        src_names[val] = key
    end
    out = ReducedCalibration{T}(dat.roi,
                                inits(T, dims, badpixvalue), #nans(T, dims),  # f
                                inits(T, dims, badpixvalue), #nans(T, dims),  # z
                                inits(T, dims, badpixvalue), #nans(T, dims),  # g
                                inits(T, dims, badpixvalue), #nans(T, dims),  # σ
                                [inits(T, dims, badpixvalue) for j in 1:nsrc], #[nans(T, dims) for j in 1:nsrc],  # s
                                src_names;
                                vpm=vpm)
    npixels = prod(dims)
    p = Progress(count(vpm); showspeed=true)
    Threads.@threads for k in 1:npixels
        vpm[k] || continue
        i = Threads.threadid()
        extract!(obj[i], dat, k)
        copyto!(x[i], xmin[i])
        x[i][2] = g
        x[i][3] = (S === :zgσs ? σ : g*σ^2)
        # try
        fit_linear_terms!(obj[i], x[i]; eta=Inf, nonnegative=nonnegative)
        vmlmb!(obj[i],x[i]; mem=n, lower=xmin[i], upper=xmax[i],  autodiff=false,
               ftol=(1e-8,0), xtol=(0,0), gtol=(0,0), maxeval=1000)
        # catch e
        #     @debug showerror(stdout, e)
        #     @debug "VMLMB crashed on pixel  $k"
        #     reset!(obj[i])
        #     continue
        # end
        out.f[k] = obj[i](x[i]) # FIXME: should not be necessary
        out.z[k] = x[i][1]
        out.g[k] = x[i][2]
        out.σ[k] = (S === :zgσs ? x[i][3] : sqrt(x[i][3]/x[i][2]))
        for j in 1:nsrc
            out.s[j][k] = x[i][j + 3]
        end
        quiet || next!(p)
    end
    return out
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
    axes(y) == (I,) || dimension_mismatch(
        "incompatible indices of destination vector")
    axes(x) == (J,) || dimension_mismatch(
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
    axes(y) == (J,) || dimension_mismatch(
        "incompatible indices of destination vector")
    axes(x) == (I,) || dimension_mismatch(
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
    fastmin(x, y)

yields the least of `x` and `y` if neither `x` nor `y` are NaNs, or `x`
otherwise.

Calling `fastmin(x,y)` is faster than `min(x,y)` but the latter propagates
NaNs (i.e., `min(x,y)` yields NaN if any of `x` or `y` is a NaN).

"""
fastmin(x::T, y::T) where {T<:Real} = (x > y ? y : x)

"""
    fastmax(x, y)

yields the greatest of `x` and `y` if neither `x` nor `y` are NaNs, or `x`
otherwise.

Calling `fastmax(x,y)` is faster than `max(x,y)` but the latter propagates
NaNs (i.e., `max(x,y)` yields NaN if any of `x` or `y` is a NaN).

"""
fastmax(x::T, y::T) where {T<:Real} = (x < y ? y : x)

end # module
