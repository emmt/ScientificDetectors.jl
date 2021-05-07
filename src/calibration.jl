module Calibration

using ..ScientificDetectors
using ..ScientificDetectors: offset, binning
import ..ScientificDetectors: regionofinterest, exposuretime

using Statistics
using OptimPackNextGen

const Colons{N} = NTuple{N,Colon}

# Union of acceptable identifer types.
const Identifiers = Union{AbstractString,Symbol,Integer}

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

`ReducedCalibration{T}` stores the calibration parameters with `T` the
floating-point type for the computations.

Constructor is called as:
    ReducedCalibration([roi,] f, z, g, σ, args...; kwds...) -> cal

where `roi` is an `N`-tuple of `DetectorAxis` describing the region of interest
(automatically guessed from argument `f` if not specified), `f` is the
co-log-likelihood, `z` is the *zero level* that is the constant bias set by the
analog to digital converter (in ADU), `g` is the detector gain (in electrons per
ADU) and `σ` is the standard deviation of the readout noise (in ADU/frame).
Arguments `f`, `z`, `g` and `σ` are pixelwise.

Additional arguments `args...` can be:

- Key-value pairs like `"cat1" => c1`, `:cat2 => c2`, ... of category
  identifiers and arrays corresponding to current terms like the dark current or
  any background flux (in ADU/second).  Arguments `c1`, `c2`, ... are assumed to
  be pixelwise.

- Two arguments: `c = [c1, c2, ...]` and `cat = ["cat1", "cat2", ...]`
  respectively a vector of current terms and of corresponding category
  identifiers.

Floating-point type, say `T`, and dimensionality, say `N`, may be specified:

    ReducedCalibration{T}([roi,] f, z, g, σ, args...; kwds...) -> cal
    ReducedCalibration{T,N}([roi,] f, z, g, σ, args...; kwds...) -> cal

Basic operations on `ReducedCalibration` instance `obj`:

    size(obj)       # yields dimensions of detector
    size(obj,k)     # yields `k`-th dimension of detector
    length(obj)     # yields number of elements of detector
    eltype(obj)     # yields floating-point type of calibration data
    T.(obj)         # convert contents of `obj` to floating-point type `T`

Other implemented methods (must be imported or prefixed by `Calibration.`):

    cologlikelihood(obj) # yields co-log-likelihood map
    detectorbias(obj)    # yields constant detector bias (in ADU)
    detectorgain(obj)    # yields detector gain (in e-/ADU)
    detectornoise(obj)   # yields standard deviation of detector noise (in ADU)
    currents(obj)        # yields all current terms
    current(obj, k)      # yields k-th current term (in ADU/s)
    categories(obj)      # yields names of current terms
    category(obj, k)     # yields name of k-th current term

"""
struct ReducedCalibration{T<:AbstractFloat,N}
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::NTuple{N,DetectorAxis}

    # Co-log-likelihood.
    f::Array{T,N}

    # Zero-level (constant bias in ADU):
    z::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise (in ADU/frame):
    σ::Array{T,N}

    # Time dependent bias, e.g. dark current and background flux, (in
    # ADU/second), may be empty or zero-filled:
    c::Vector{Array{T,N}}

    # Categories of the different sources responsible of the different
    # time-dependent bias terms.
    cat::Vector{String}

    # Inner constructor provided to force using outer constructors.
    function ReducedCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                     f::Array{T,N},
                                     z::Array{T,N},
                                     g::Array{T,N},
                                     σ::Array{T,N},
                                     c::Vector{Array{T,N}},
                                     cat::Vector{String};
                                     check::Bool = false
                                     ) where {T<:AbstractFloat,N}
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(f) == dims
        @assert size(z) == dims
        @assert size(g) == dims
        @assert size(σ) == dims
        @assert length(cat) == length(c)
        for k ∈ eachindex(c)
            @assert size(c[k]) == dims
        end
        obj = new{T,N}(roi, f, z, g, σ, c, cat)
        check && checkvalues(obj)
        return obj
    end
end

#
# Simple outer constructors (mostly for conversion).  Note that a constructor
# of an immutable structure can safely return its argument.
#
ReducedCalibration(obj::ReducedCalibration) = obj
function ReducedCalibration(roi::NTuple{N,DetectorAxis},
                            f::AbstractArray{<:Real,N},
                            z::AbstractArray{<:Real,N},
                            g::AbstractArray{<:Real,N},
                            σ::AbstractArray{<:Real,N},
                            c::AbstractVector{Array{<:Real,N}},
                            cat::AbstractVector{String};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ),
                           map(eltype, c)...))
    ReducedCalibration{T}(roi, f, z, g, σ, c, cat; kwds...)
end

ReducedCalibration{T}(obj::ReducedCalibration{T}) where {T} = obj
ReducedCalibration{T}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T}(obj.roi, obj.f, obj.z, obj.g, obj.σ, obj.c, obj.cat)
function ReducedCalibration{T}(roi::NTuple{N,DetectorAxis},
                               f::AbstractArray{<:Real,N},
                               z::AbstractArray{<:Real,N},
                               g::AbstractArray{<:Real,N},
                               σ::AbstractArray{<:Real,N},
                               c::AbstractVector{Array{<:Real,N}},
                               cat::AbstractVector{String};
                               kwds...) where {T<:AbstractFloat,N}
    # Call the inner constructor with all arguments of correct type.
    ReducedCalibration{T,N}(roi,
                            convert(Array{T,N}, f),
                            convert(Array{T,N}, z),
                            convert(Array{T,N}, g),
                            convert(Array{T,N}, σ),
                            map(x -> convert(Array{T,N}, x), c),
                            convert(Array{String}, cat);
                            kwds...)
end

ReducedCalibration{T,N}(obj::ReducedCalibration{T,N}) where {T,N} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T}(obj)
function ReducedCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                 f::AbstractArray{<:Real,N},
                                 z::AbstractArray{<:Real,N},
                                 g::AbstractArray{<:Real,N},
                                 σ::AbstractArray{<:Real,N},
                                 c::AbstractVector{Array{<:Real,N}},
                                 cat::AbstractVector{String};
                                 kwds...) where {T<:AbstractFloat,N}
    ReducedCalibration{T}(roi, f, z, g, σ, c, cat; kwds...)
end

#
# Getters.
#
regionofinterest(obj::ReducedCalibration) = obj.roi
cologlikelihood(obj::ReducedCalibration) = obj.f
detectorbias(obj::ReducedCalibration) = obj.z
detectorgain(obj::ReducedCalibration) = obj.g
detectornoise(obj::ReducedCalibration) = obj.σ
currents(obj::ReducedCalibration) = obj.c
current(obj::ReducedCalibration, k::Integer) = getindex(currents(obj), k)
categories(obj::ReducedCalibration) = obj.cat
category(obj::ReducedCalibration, k::Integer) = getindex(categories(obj), k)

#
# Basic operations on ReducedCalibration structure.
#
Base.eltype(::ReducedCalibration{T}) where {T} = T
Base.size(obj::ReducedCalibration) = size(regionofinterest(obj))
Base.size(obj::ReducedCalibration, i) = size(regionofinterest(obj), i)
Base.length(obj::ReducedCalibration) = prod(size(obj))
Base.convert(::Type{T}, obj::ReducedCalibration) where {T<:ReducedCalibration} =
    T(obj)

Base.show(io::IO, obj::ReducedCalibration{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " ReducedCalibration{$T,$N}:")
    for i in 1:length(categories(obj))
        print(io, "\n - cat", i, ": \"", identifier(category(obj,i)), "\"")
    end
end

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::ReducedCalibration) where {T<:AbstractFloat} =
    ReducedCalibration{T}(obj)

#
# More complex outer constructors for ReducedCalibration structure.
#

# Provide a ROI if not specified and parse current terms.
function ReducedCalibration(f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(map(DetectorAxis, size(f)), f, z, g, σ,
                       _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T}(f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(map(DetectorAxis, size(f)), f, z, g, σ,
                          _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T,N}(f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(map(DetectorAxis, size(f)), f, z, g, σ,
                            _getcurrents(args...)...; kwds...)
end

# Parse current terms.
function ReducedCalibration(roi::Tuple{Vararg{DetectorAxis}},
                            f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T}(roi::Tuple{Vararg{DetectorAxis}},
                               f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration(roi::NTuple{N,DetectorAxis},
                            f::AbstractArray,
                            z::AbstractArray,
                            g::AbstractArray,
                            σ::AbstractArray,
                            c::AbstractVector{<:AbstractArray},
                            cat::AbstractVector{<:Identifiers};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ),
                           _promote_eltype(c)))
    ReducedCalibration{T,N}(roi, f, z, g, σ, c, cat; kwds...)
end

function ReducedCalibration{T}(roi::NTuple{N,DetectorAxis},
                               f::AbstractArray,
                               z::AbstractArray,
                               g::AbstractArray,
                               σ::AbstractArray,
                               c::AbstractVector{<:AbstractArray},
                               cat::AbstractVector{<:Identifiers};
                               kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, c, cat; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray,
                                 z::AbstractArray,
                                 g::AbstractArray,
                                 σ::AbstractArray,
                                 c::AbstractVector{<:AbstractArray},
                                 cat::AbstractVector{<:Identifiers};
                                 kwds...) where {T,N}
    T <: AbstractFloat || error("parameter `T` must be a floating-point type")
    length(roi) == N || error("ROI has incompatible number of dimensions")
    length(cat) == length(c) || error("incompatible number of categories")
    dims = size(roi)

    function fixarray(A::AbstractArray)
        Base.has_offset_axes(A) && error("array has non-standard indexing")
        eltype(A) <: Real || error("array has incompatible element type")
        ndims(A) == N || error("array has incompatible number of dimensions")
        size(A) == dims ||
            throw(DimensionMismatch("array has incompatible dimensions"))
        return convert(Array{T,N}, A)
    end
    ReducedCalibration{T,N}(roi,
                            fixarray(f),
                            fixarray(z),
                            fixarray(g),
                            fixarray(σ),
                            map(fixarray, c),
                            map(identifier, cat); kwds...)
end

# Convert pairs like "key1"=>arr1, :key2=>arr2, ... in a list of
# arrays and a list of identifiers.
_getcurrents(args::Pair{<:Union{AbstractString,Symbol},<:AbstractArray}...) =
    (collect(map(x -> x[2], args)),
     collect(map(x -> identifier(x[1]), args)))
_getcurrents() = Int8[], String[]
_getcurrents(c::AbstractVector{<:AbstractArray}, cat::AbstractVector) =
    (c, cat)

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

"""
    checkvalues(obj)

throws an error if some values in the reduced calibration object `obj` are
invalid.

"""
function checkvalues(cal::ReducedCalibration)
    f, z, g, σ, c = cal.f, cal.z, cal.g, cal.σ, cal.c
    dims = size(cal)
    @assert size(f) == dims
    @assert size(z) == dims
    @assert size(g) == dims
    @assert size(σ) == dims
    for k ∈ eachindex(c)
        @assert size(c[k]) == dims
        all(x -> isfinite(x) && x ≥ 0, c[k]) ||
            error("some invalid values in time-dependent bias")
    end
    all(x -> isfinite(x), z) ||
        error("some invalid values in constant bias")
    all(x -> isfinite(x) && x ≥ 0, g) ||
        error("some invalid values in detector gain")
    all(x -> isfinite(x) && x ≥ 0, σ) ||
        error("some invalid values in readout noise")
end

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
part of the union `Identifiers` (strings, symbols or integers).

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
    c::Vector{T} # contributions of the different sources
end

# Structure used to store all calibration data.
"""
    CalibrationData(D, keys, Δt) -> obj

yields an object which stores detector calibration data.  Argument `D` is a
vector of detector data frames, `keys` and `Δt` respectively specify the
identifier and exposure time of the corresponding data frame.  The keys can be
integers, symbols or strings.  A given key uniquely identify the category of
the corresponding data frame. Exposure times are in seconds.

    numberofdataframes(obj)  # yields the number of data frames
    numberofcategories(obj)  # yields the number of different categories
    dataframes(obj)          # yields the vector of data frames
    dataframe(obj, i)        # yields the i-th data frame
    categories(obj)          # yields the category indices of the data frames
    category(obj, i)         # yields the category index of the i-th data frame
    exposuretimes(obj)       # yields the exposure times of the data frames
    exposuretime(obj, i)     # yields the exposure time of the i-th data frame
    uniqueidentifiers(obj)   # yields the list of unique identifiers of categories
    uniqueidentifier(obj, l) # yields the l-th unique identifier of categories

"""
struct CalibrationData{P<:Real,N,T<:AbstractFloat}
    dims::Dims{N}            # dimensions of frames
    data::Vector{Array{P,N}} # data[i][j] is j-th pixel of i-th frame
    Δt::Vector{T}            # Δt[i] yields the exposure time of i-th frame
    cat::Vector{Int}         # cat[i] yields the category index of i-th frame
    uid::Vector{String}      # uid[l] is the unique identifer of l-th
                             # calibration category
    function CalibrationData{P,N,T}(data::AbstractVector{Array{P,N}},
                                    keys::AbstractVector{<:Identifiers},
                                    Δt::AbstractVector{T}
                                    ) where {P<:Real,N,T<:AbstractFloat}
        nframes = length(data)
        @assert nframes > 0
        @assert length(keys) == nframes
        @assert length(Δt) == nframes
        dims = size(first(data))
        for A in data
            @assert size(A) == dims
        end
        @assert minimum(Δt) ≥ 0

        cat, uid = uniquecategories(keys)
        ntypes = maximum(cat)

        return new{P,N,T}(dims,
                          convert(Vector{Array{P,N}}, data),
                          convert(Vector{T}, Δt), cat, uid)
    end
end

Base.size(obj::CalibrationData) = obj.dims
Base.size(obj::CalibrationData{P,N,T}, d::Integer) where {P,N,T} =
    (d < 1 ? error("invalid dimension index") :
     d ≤ N ? size(obj)[d] : 1)

dataframes(obj::CalibrationData) = obj.data
categories(obj::CalibrationData) = obj.cat
exposuretimes(obj::CalibrationData) = obj.Δt
uniqueidentifiers(obj::CalibrationData) = obj.uid

numberofdataframes(obj::CalibrationData) = length(dataframes(obj))
numberofcategories(obj::CalibrationData) = length(uniqueidentifiers(obj))

dataframe(obj::CalibrationData, i::Integer) = getindex(dataframes(obj), i)
category(obj::CalibrationData, i::Integer) = getindex(categories(obj), i)
exposuretime(obj::CalibrationData, i::Integer) = getindex(exposuretime(obj), i)
uniqueidentifier(obj::CalibrationData, l::Integer) =
    getindex(uniqueidentifiers(obj), l)

"""
    ReducedCalibration(cal) -> redcal

fit the detector parameters in calibration data `cal`.

"""
function ReducedCalibration(cal::CalibrationData{P,N,T}) where {P,N,T}
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
    function fg!(x::Vector{T}, gx::Vector{T})
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
            copyto!(res.c, c)
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


#------------------------------------------------------------------------------
# SIMPLE CALIBRATION BASED ON AVERAGED IMAGES

struct SimpleCalibration{T<:AbstractFloat,N}
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::NTuple{N,DetectorAxis}

    # Exposure time (in seconds).
    Δt::Float64

    # Co-log-likelihood.
    f::Array{T,N}

    # Amplitude correction factor (in flux units per ADU):
    a::Array{T,N}

    # Bias correction (in ADU):
    b::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise plus background (in ADU/frame):
    σ::Array{T,N}

    # Inner constructor provided to force using outer constructors.
    function SimpleCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                    Δt::Real,
                                    f::Array{T,N},
                                    a::Array{T,N},
                                    b::Array{T,N},
                                    g::Array{T,N},
                                    σ::Array{T,N}) where {T<:AbstractFloat,N}
        @assert isfinite(Δt) && Δt ≥ 0
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(f) == dims
        @assert size(a) == dims
        @assert size(b) == dims
        @assert size(g) == dims
        @assert size(σ) == dims
        return new{T,N}(roi, Δt, f, a, b, g, σ)
    end
end

#
# Simple outer constructors for conversion.
#
SimpleCalibration(obj::SimpleCalibration) = obj
function SimpleCalibration(roi::NTuple{N,DetectorAxis},
                           Δt::Real,
                           f::AbstractArray{<:Real,N},
                           a::AbstractArray{<:Real,N},
                           b::AbstractArray{<:Real,N},
                           g::AbstractArray{<:Real,N},
                           σ::AbstractArray{<:Real,N}) where {N}
    T = float(promote_type(eltype(f), eltype(a), eltype(b), eltype(g), eltype(σ)))
    SimpleCalibration{T}(roi, Δt, f, a, b, g, σ)
end

SimpleCalibration{T}(obj::SimpleCalibration{T}) where {T} = obj
SimpleCalibration{T}(obj::SimpleCalibration{<:Any,N}) where {T<:AbstractFloat,N} =
    SimpleCalibration{T}(obj.roi, obj.Δt, obj.f, obj.a, obj.b, obj.g, obj.σ)
function SimpleCalibration{T}(roi::NTuple{N,DetectorAxis},
                              Δt::Real,
                              f::AbstractArray{<:Real,N},
                              a::AbstractArray{<:Real,N},
                              b::AbstractArray{<:Real,N},
                              g::AbstractArray{<:Real,N},
                              σ::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    # Call the inner constructor with all arguments of correct type.
    SimpleCalibration{T,N}(roi, Δt,
                           convert(Array{T,N}, f),
                           convert(Array{T,N}, a),
                           convert(Array{T,N}, b),
                           convert(Array{T,N}, g),
                           convert(Array{T,N}, σ))
end

SimpleCalibration{T,N}(obj::SimpleCalibration{T,N}) where {T,N} = obj
SimpleCalibration{T,N}(obj::SimpleCalibration{<:Any,N}) where {T,N} =
    SimpleCalibration{T}(obj)
function SimpleCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                Δt::Real,
                                f::AbstractArray{<:Real,N},
                                a::AbstractArray{<:Real,N},
                                b::AbstractArray{<:Real,N},
                                g::AbstractArray{<:Real,N},
                                σ::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    SimpleCalibration{T}(roi, Δt, f, a, b, g, σ)
end

#
# Getters.
#
regionofinterest(obj::SimpleCalibration) = obj.roi
exposuretime(obj::SimpleCalibration) = obj.Δt


#
# Basic operations on SimpleCalibration structure.
#
Base.eltype(::SimpleCalibration{T}) where {T} = T
Base.size(obj::SimpleCalibration) = size(regionofinterest(obj))
Base.size(obj::SimpleCalibration, i) = size(regionofinterest(obj), i)
Base.length(obj::SimpleCalibration) = prod(size(obj))
Base.convert(::Type{T}, obj::SimpleCalibration) where {T<:SimpleCalibration} =
    T(obj)

function Base.show(io::IO, obj::SimpleCalibration{T,N}) where {T,N}
    print(io, "SimpleCalibration{$T,$N}: size = ")
    join(io, size(obj),"×")
    print(io, ", Δt = ", exposuretime(obj), " s")
end

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::SimpleCalibration) where {T<:AbstractFloat} =
    SimpleCalibration{T}(obj)

"""
    SimpleCalibration{T}(ROI, Δt,
                         NumDark, AvgDark, VarDark,
                         NumLamp, AvgLamp, VarLamp,
                         AvgFlat)

yields reduced calibration data given sample means and variances of 3 kinds of
images: "dark" (or "bias") images, "lamp" images with a stable illumination
(although not necessarily uniform) and "flat" images with a uniform
illumination.  Arguments are as follows:

* `ROI` is a `N`-tuple of `DetectorAxis` describing the region of interest;
* `Δt` is the exposure time (in seconds);
* `NumDark` is the number of averaged "dark" images;
* `AvgDark` is the sample mean of the "dark" images;
* `VarDark` is the sample variance of the "dark" images;
* `NumLamp` is the number of averaged "lamp" images;
* `AvgLamp` is the sample mean of the "lamp" images;
* `VarLamp` is the sample variance of the "lamp" images;
* `AvgFlat` is the sample mean of the "flat" images;

The sample variances are the maximum likelihood (i.e. biased) estimator of the
variances.  The sample mean and variance are computed as follows:

    avg = (1/N)*(x1 + x2 + ... + xN)
    var = (1/N)*((x1 - avg)^2 + (x2 - avg)^2 + ... + (xN - avg)^2)

Set keyword `unbiased` to `true` if the unbiased variances are provided, that
is computed as:

    var = (1/(N - 1))*((x1 - avg)^2 + (x2 - avg)^2 + ... + (xN - avg)^2)

See also [`ReducedCalibration`](@ref).

"""
function SimpleCalibration{T}(roi::NTuple{N,DetectorAxis},
                              Δt::Real,
                              NumDark::Integer,
                              AvgDark::AbstractArray{<:Real,N},
                              VarDark::AbstractArray{<:Real,N},
                              NumLamp::Integer,
                              AvgLamp::AbstractArray{<:Real,N},
                              VarLamp::AbstractArray{<:Real,N},
                              AvgFlat::AbstractArray{<:Real,N};
                              optimal::Bool = false,
                              unbiased::Bool=false,
                              umin::Real=1e-20) where {T<:AbstractFloat,N}
    # Check arguments.
    @assert isfinite(Δt) && Δt > 0
    @assert isfinite(umin) && umin > 0
    dims = size(roi)
    @assert !Base.has_offset_axes(AvgDark, VarDark, AvgLamp, VarLamp, AvgFlat)
    @assert NumDark > 1
    @assert size(AvgDark) == dims
    @assert size(VarDark) == dims
    @assert NumLamp > 1
    @assert size(AvgLamp) == dims
    @assert size(VarLamp) == dims
    @assert size(AvgFlat) == dims

    # Local variables fopr computations.
    Ndark = Float64(NumDark)
    Nlamp = Float64(NumLamp)
    local Mdark::Float64, Vdark::Float64
    local Mlamp::Float64, Vlamp::Float64
    local Mflat::Float64

    # Allocate arrays for the result.
    f = Array{T}(undef, dims)
    a = Array{T}(undef, dims)
    b = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    σ = Array{T}(undef, dims)
    obj = SimpleCalibration{T,N}(roi, Δt, f, a, b, g, σ)

    # Perform a first pass to check the values and initialize parameters to
    # their sub-optimal estimators.
    checkvalues(Mdark, Vdark, Mlamp, Vlamp, Mflat) =
        ((isfinite(Mdark) & isfinite(Vdark) &
          isfinite(Mlamp) & isfinite(Vlamp) &
          isfinite(Mflat)) && 0 < Vdark < Vlamp &&
         Mdark < min(Mlamp, Mflat))

    # Variance correction factors for unbiased estimators.
    γdark = (unbiased ? one(Float64) : Ndark/(Ndark - 1))
    γlamp = (unbiased ? one(Float64) : Nlamp/(Nlamp - 1))

    for j in eachindex(f, a, b, g, σ, AvgDark, VarDark, AvgLamp, VarLamp)
        Mdark = Float64(AvgDark[j])
        Vdark = Float64(VarDark[j])*γdark
        Mlamp = Float64(AvgLamp[j])
        Vlamp = Float64(VarLamp[j])*γlamp
        Mflat = Float64(AvgFlat[j])
        if checkvalues(Mdark, Vdark, Mlamp, Vlamp, Mflat)
            f[j] = (Ndark*(1 + log(Vdark)) - 1) + (Nlamp*(1 + log(Vlamp)) - 1)
            a[j] = 1/(Mflat - Mdark)
            b[j] = Mdark
            g[j] = (Mlamp - Mdark)/(Vlamp - Vdark)
            σ[j] = sqrt(Vdark)
        else
            f[j] = Inf
            a[j] = 0
            b[j] = 0
            g[j] = 0
            σ[j] = Inf
        end
    end

    # Unless sub-optimal estimators have been requested, perform a second pass
    # to obtain optimal estimators.
    if optimal == false
        return obj
    end

    # Allocate workspace.  Computations are done in double precision.
    var = Vector{Float64}(undef, 2)
    #grd = Vector{Float64}(undef, 2)
    lim = Vector{Float64}(undef, 2)

    # Workspace to store the best solution so far.
    best = Vector{Float64}(undef, 5)

    # Objective function to minimize (as a closure).
    function fg!(var::Vector{Float64}, grd::Vector{Float64})
        local f, b, g, x, y
        # Extract parameters.
        @assert length(var) == length(grd) == 2
        x = var[1]
        y = var[2]

        # Weights.
        Wdark = Ndark/x
        Wlamp = Nlamp/(x + y)

        # Best bias.
        b = (Wdark*Mdark + Wlamp*(Mlamp - y))/(Wdark + Wlamp)

        # Residuals.
        Rdark = Mdark - b
        Rlamp = Mlamp - (b + y)

        # Best gain.
        g = (Ndark + Nlamp)/(Wdark*(Vdark + Rdark^2) + Wlamp*(Vlamp + Rlamp^2))

        # Gradient.
        gtemp = Wlamp*(1  - g*(Vlamp + Rlamp^2)/(x + y))
        grd[1] = gtemp + Wdark*(1  - g*(Vdark + Rdark^2)/x)
        grd[2] = gtemp - 2*Wlamp*g*Rlamp

        # Objective function.
        f = Ndark*(1 + log(x/g)) + Nlamp*(1 + log((x + y)/g))
        if f < best[1]
            best[1] = f
            best[2] = b
            best[3] = g
            best[4] = x
            best[5] = y
        end
        return f
    end

    # Variance correction factors for max. likelihood estimators.
    γdark = (unbiased ? (Ndark - 1)/Ndark : one(Float64))
    γlamp = (unbiased ? (Nlamp - 1)/Nlamp : one(Float64))

    for j in eachindex(f, a, b, g, σ, AvgDark, VarDark, AvgLamp, VarLamp)
        Mdark = Float64(AvgDark[j])
        Vdark = Float64(VarDark[j])*γdark
        Mlamp = Float64(AvgLamp[j])
        Vlamp = Float64(VarLamp[j])*γlamp
        Mflat = Float64(AvgFlat[j])
        if a[j] > 0
            best[1] = Inf
            best[2] = NaN
            best[3] = NaN
            best[4] = NaN
            best[5] = NaN
            # Initial solution:
            #     b  = Mdark
            #     g  = (Mlamp - Mdark)/(Vlamp - Vdark)
            #     x  = g*Vdark
            #     y  = Mlamp - Mdark
            var[1] = (Mlamp - Mdark)/(Vlamp/Vdark - 1) # initial x
            var[2] = Mlamp - Mdark                     # initial y
            lim[1] = var[1]/100
            lim[2] = 0

            vmlmb!(fg!, var, mem=2, lower=lim)
            if Mflat > best[2]
                f[j] = best[1]
                a[j] = 1/(Mflat - best[2])
                b[j] = best[2]
                g[j] = best[3]
                σ[j] = sqrt(best[4]/g[j])
            end
        end
    end

    return obj
end

end # module
