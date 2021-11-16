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

struct FitWorkspace{T<:AbstractFloat}
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
    wrk = FitWorkspace{T=float(eltype(H))}(H, nsub)

yields a workspace for fitting the parameters of the model of a detector pixel
for sources to currents matrix `H` and `nsub` data subsets.  All entries of `H`
must be nonnegative (this is checked).  The same workspace can be re-used for
another pixel provided the matrix `H` remains the same.

"""
FitWorkspace(H::AbstractMatrix{<:Real}, nsub::Integer) =
    FitWorkspace{float(eltype(H))}(H, nsub)

function FitWorkspace{T}(H::AbstractMatrix{<:Real},
                         nsub::Integer) where {T<:AbstractFloat}
    isnonnegative(H) || error(
        "entries of sources to currents matrix must be nonnegative")
    nrows, ncols = size(H)
    n = ncols + 1
    return FitWorkspace{T}(
        H,
        Vector{T}(  undef, nrows), # c
        Vector{T}(  undef, nrows), # ∂c
        Vector{T}(  undef, nsub),  # Δt
        Vector{Int}(undef, nsub),  # l
        Vector{Int}(undef, nsub),  # n
        Vector{T}(  undef, nsub),  # avg
        Vector{T}(  undef, nsub),  # var
        Vector{T}(  undef, nrows), # workspace w1
        Vector{T}(  undef, nrows), # workspace w2
        Vector{T}(  undef, nrows), # workspace w3
        Vector{T}(  undef, nrows), # workspace w4
        Matrix{T}(  undef, n, n),  # LHS matrix A
        Vector{T}(  undef, n))     # RHS vector b
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

# Return the number of data subsets in fit workspace.
function Base.length(wrk::FitWorkspace;
                     checksizes::Bool=false,
                     checkindices::Bool=false)
    nsub = length(wrk.l) # number of subsets
    if checksizes || checkindices
        nrows, ncols = size(wrk.H)
        n = ncols + 1
        length(wrk.c)   == nrows || error("bad number of categories")
        length(wrk.∂c)  == nrows || error("bad number of category gradients")
        length(wrk.Δt)  == nsub  || error("bad number of exposure times")
        length(wrk.avg) == nsub  || error("bad number of empirical means")
        length(wrk.var) == nsub  || error("bad number of empirical variances")
        length(wrk.n)   == nsub  || error("bad number of subset sizes")
        length(wrk.w1)  == nrows || error("bad size for workspace W1")
        length(wrk.w2)  == nrows || error("bad size for workspace W2")
        length(wrk.w3)  == nrows || error("bad size for workspace W3")
        length(wrk.w4)  == nrows || error("bad size for workspace W4")
        size(wrk.A)     == (n,n) || error("bad size for LHS matrix A")
        length(wrk.b)   == n     || error("bad size for RHS vector b")
        if checkindices
            flag = true
            @inbounds @simd for i in eachindex(wrk.l)
                flag &= ((wrk.l[i] - 1)%UInt < nrows)
            end
            flag || error("out of bound category index")
        end
    end
    return nsub
end

"""
    extract!(wrk, cal, k) -> wrk

extracts into workspace `wrk` the data for `k`-th pixel in calibration data
`cal`.

"""
function extract!(wrk::FitWorkspace,
                  cal::CalibrationData{T,N},
                  k::Integer) where {T,N}
    nrows, ncols = size(cal.src_to_cat)
    nsub = length(wrk; checksizes=true)
    length(cal.stat) == nsub || error(
        "fit workspace assumes a different number of subsets")
    @inbounds for (key, i) ∈ cal.stat_index
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
                           x::Vector{T} = zeros(T, size(wrk.H,2) + 1);
                           reset::Bool = false,
                           nonnegative::Bool = true,
                           eta::Real = Inf,
                           mem::Integer = 5,
                           kwds...) where {T<:AbstractFloat}
    # Do a weighted least squares fit on all the linear parameters with the
    # positivity constraint on the source terms.
    eq = form_normal_equations!(wrk, x, to_type(T, eta))
    reset && fill!(x, 0) # FIXME:
    if nonnegative
        # Solve the normal equations under the constraints that the source
        # terms are nonnegative.
        n = length(x)
        xmin = Vector{T}(undef, n) # FIXME: make is part of wrk
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
    form_normal_equations!(wrk, x, η=Inf) -> eq

forms the normal equations for fitting the flux terms (the sources and the
zero-level).  Argument `η = g⋅σ² > 0` is the gain time the variance of the
read-out noise.  The weights are computed as:

    w[k,l] = n[k,l]/(c[l]⋅Δt[k,l] + η)

where `c = H*x[1:end-1]` is the flux per category of calibration with `H =
wrk.H` the sources to categories matrix, `x[1:end-1]` the fluxes of the
sources, and `x[end]` the zero-level `z` (not used here).

If `η = Inf`, then `x` is not used and the weights are assumed to be given by
the number of samples: ` w[k,l] = n[k,l]`.

"""
function form_normal_equations!(wrk::FitWorkspace{T},
                                x::AbstractVector{T},
                                η::Real = Inf) where {T<:AbstractFloat}
    return form_normal_equations!(wrk, x, to_type(T, η))
end

function form_normal_equations!(wrk::FitWorkspace{T},
                                x::AbstractVector{T},
                                η::T) where {T<:AbstractFloat}
    # Extract parameters from workspace.
    nsub = length(wrk; checksizes=true, checkindices=true)
    H = wrk.H
    nrows, ncols = size(H)
    n = ncols + 1
    length(x) == n || error("variables must have ", n, " elements")
    J = Base.OneTo(ncols)
    L = Base.OneTo(nrows)

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
    #    d[k,l] = wrk.avg[k,l]
    #    w[k,l] = n[k,l]/(c[l]*Δt[k,l] + η)
    #
    # the "data" and the weights.  The sum is over index `i ~ (k,l)` such that
    # category index is given by `l = wrk.l[i]`.
    #
    w1 = fill!(wrk.w1, zero(T))
    w2 = fill!(wrk.w2, zero(T))
    w3 = fill!(wrk.w3, zero(T))
    Ann = zero(T)
    bn = zero(T)
    if reweighted
        # Compute fluxes in calibration categories and flux-dependent weights.
        c = mvmult!(wrk.c, H, view(x, 1:ncols))
        @inbounds for i in 1:nsub
            Δt  = wrk.Δt[i]
            l   = wrk.l[i]
            cΔt = c[l]*Δt
            w   = T(wrk.n[i])/(cΔt + η)
            d   = wrk.avg[i]
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
            Δt  = wrk.Δt[i]
            l   = wrk.l[i]
            w   = T(wrk.n[i])
            d   = wrk.avg[i]
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
        A = wrk.A
        w4 = wrk.w4 # FIXME: wrk.c could be used as a temporary workspace here
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
        b = wrk.b
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

"""
    f = objfunc(wrk, z, g, η, s)

yields the value of the objective function associated with workspace `wrk` and
for model parameters `x = (z, g, η, s...)` with `z` the zero level, `g` the
gain, `η` the variance of the readout noise times the gain and the source terms
`s`.

"""
function objfunc(wrk::FitWorkspace{T},
                 z::Real, g::Real, η::Real,
                 s::AbstractVector{T}) where {T<:AbstractFloat}
    return objfunc(wrk, to_type(T, z), to_type(T, g), to_type(T, η), s)
end

function objfunc(wrk::FitWorkspace{T},
                 z::T, g::T, η::T,
                 s::AbstractVector{T}) where {T<:AbstractFloat}
    check_objfunc_args(g, η, s)
    c = mvmult!(wrk.c, wrk.H, s) # compute fluxes in calibration categories
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    nsub = length(wrk; checksizes=true, checkindices=true)
    @inbounds @simd for i ∈ 1:nsub
        Δt  = wrk.Δt[i]     # exposure time
        l   = wrk.l[i]      # category index
        n   = T(wrk.n[i])   # number of samples in subset
        cΔt = c[l]*Δt       # contribution of sources
        d   = wrk.avg[i]    # data = sample mean
        r   = (cΔt + z) - d # residuals: model - data
        w   = n/(cΔt + η)   # weight
        χ² += w*(wrk.var[i] + r^2)
        sum_n_logw_n += n*log(w/n)
        N += n
    end
    return g*χ² - sum_n_logw_n - N*log(g)
end

function check_objfunc_args(g::Real, η::Real, s::AbstractVector{<:Real})
    g > 0 || throw_argument_error("gain `g` must be positive")
    η > 0 || throw_argument_error("readout variance `η` must be positive")
    isnonnegative(s) || throw_argument_error(
        "source terms `s` must be nonnegative")
end

"""
    f = objfunc!(wrk, z, g, η, s, grd)

yields the value of the objective function `f(x)` associated with workspace
`wrk` and overwrites `grd` with the gradient `∇f(x)` for model parameters `x =
(z, g, η, s...)` with `z` the zero level, `g` the gain, `η` the variance of the
readout noise times the gain and the source terms `s`.

"""
function objfunc!(wrk::FitWorkspace{T},
                  z::Real, g::Real, η::Real,
                  s::AbstractVector{T},
                  grd::AbstractVector{T}) where {T<:AbstractFloat}
    return objfunc!(wrk, to_type(T, z)::T, to_type(T, g)::T, to_type(T, η)::T,
                    s, grd)
end

function objfunc!(wrk::FitWorkspace{T},
                  z::T, g::T, η::T,
                  s::AbstractVector{T},
                  grd::AbstractVector{T}) where {T<:AbstractFloat}
    check_objfunc_args(g, η, s)
    length(grd) == 3 + length(s) || error("bad gradient size")
    c = mvmult!(wrk.c, wrk.H, s) # compute fluxes in calibration categories
    N = 0 # to count total number of data
    χ² = zero(T) # to sum χ²/g terms
    sum_n_logw_n = zero(T) # to sum n⋅log(w/n)
    ∂c = fill!(wrk.∂c, 0) # to compute ∂L/∂c
    ∂z = zero(T) # to compute ∂L/∂z
    ∂η = zero(T) # to compute ∂L/∂η
    nsub = length(wrk; checksizes=true, checkindices=true)
    @inbounds @simd for i ∈ 1:nsub
        Δt  = wrk.Δt[i]     # exposure time
        l   = wrk.l[i]      # category index
        n   = T(wrk.n[i])   # number of samples in subset
        cΔt = c[l]*Δt       # contribution of sources
        d   = wrk.avg[i]    # data = sample mean
        r   = (cΔt + z) - d # residuals: model - data
        q   = cΔt + η       # model variance times gain
        w   = n/q           # weight
        v   = wrk.var[i] + r^2
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
    mvmult!(view(grd, 4:length(grd)), wrk.H', ∂c)
    # Return total objective function.
    return g*χ² - sum_n_logw_n - N*log(g)
end

"""

    x[1]     = z # bias or zero-level (ADU)
    x[2]     = g # gain (e-/ADU)
    x[3]     = σ # standard deviation of read-out noise (ADU)
    x[4:end] = s # source terms (ADU/s)

    fg!(x, g) = ScientificDetectors.Calibration.objfunc!(wrk, x, g)
    pmin = zeros(T, length(x))
    pmin[1] = -Inf       # min. for z
    pmin[2] = 1.0        # min. for g
    pmin[3] = sqrt(1/12) # min. for σ
    vmlmb(fg!, copy(p); lower=pmin, verb=1, mem=length(x), maxiter=1000,
          maxeval=5000, ftol=(0,0), xtol=(0,0), gtol=(1e-5,0), autodiff=false)

"""
function objfunc!(wrk::FitWorkspace{T},
                  x::Vector{T},
                  grd::Vector{T}) where {T<:AbstractFloat}
    nsub = length(wrk; checksizes=true, checkindices=true)
    inds = axes(x)
    axes(grd) == inds || error(
        "variables and gradients have different indices")
    first(inds[1]) == 1 || error(
        "variables have non-standard indexing")
    xlen, ncols = length(x), size(wrk.H, 2)
    xlen == ncols + 3 ||  error(
        "variables must have ", ncols + 3, " elements, got ", xlen)
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
        c = mvmult!(wrk.c, wrk.H, view(x, I))

        # Loop to integrate objective function and its gradient.
        f      = zero(T)          # to compute f
        ∂f_∂z  = zero(T)          # to compute ∂f/∂z
        ∂f_∂ρ  = zero(T)          # to compute ∂f/∂ρ
        ∂f_∂σ² = zero(T)          # to compute ∂f/∂σ²
        ∂f_∂c  = fill!(wrk.∂c, 0) # to compute ∂f/∂c
        @simd for i ∈ 1:nsub
            # Objective function for this subset.
            Δt  = wrk.Δt[i]        # exposure time
            l   = wrk.l[i]         # category index
            n   = T(wrk.n[i])      # number of samples in subset
            cΔt = c[l]*Δt          # contribution of sources
            m   = cΔt + z          # model of data
            r   = m - wrk.avg[i]   # residuals = model - sample mean
            q   = r^2 + wrk.var[i] # quadratic error
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
        mvmult!(view(grd, I), wrk.H', ∂f_∂c)
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
