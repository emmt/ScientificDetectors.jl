"""
    ReducedCalibration{T}([roi,] f, z, g, σ, args...; kwds...) -> cal

builds an instance of `ReducedCalibration{T}` to store the calibration
parameters with `T` the floating-point type for the computations and where
`roi` is an `N`-tuple of `DetectorAxis` describing the region of interest
(automatically guessed from argument `f` if not specified), `f` is the
co-log-likelihood, `z` is the *zero level* that is the constant bias set by the
analog to digital converter (in ADU), `g` is the detector gain (in electrons
per ADU) and `σ` is the standard deviation of the readout noise (in ADU).
Arguments `f`, `z`, `g` and `σ` are pixelwise.

Additional arguments `args...` can be:

- Key-value pairs like `"src1" => c1`, `:src2 => c2`, ... of category
  identifiers and arrays corresponding to current terms like the dark current or
  any background flux (in ADU/second).  Arguments `c1`, `c2`, ... are assumed to
  be pixelwise.

- Two arguments: `s = [c1, c2, ...]` and `src = ["src1", "src2", ...]`
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
    s::Vector{Array{T,N}}

    # Categories of the different sources responsible of the different
    # time-dependent bias terms.
    src::Vector{String}

    # Inner constructor provided to force using outer constructors.
    function ReducedCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                     f::AbstractArray{T,N},
                                     z::AbstractArray{T,N},
                                     g::AbstractArray{T,N},
                                     σ::AbstractArray{T,N},
                                     s::AbstractVector{<:AbstractArray{T,N}},
                                     src::AbstractVector{<:AbstractString};
                                     check::Bool = false
                                     ) where {T<:AbstractFloat,N}
        checkindices(ReducedCalibration, roi, f, z, g, σ, s, src)
        obj = new{T,N}(roi, f, z, g, σ, s, src)
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
                            s::AbstractVector{<:AbstractArray{<:Real,N}},
                            src::AbstractVector{<:AbstractString};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ),
                           map(eltype, s)...))
    ReducedCalibration{T}(roi, f, z, g, σ, s, src; kwds...)
end

ReducedCalibration{T}(obj::ReducedCalibration{T}) where {T} = obj
ReducedCalibration{T}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T,N}(obj)
function ReducedCalibration{T}(roi::NTuple{N,DetectorAxis},
                               f::AbstractArray{<:Real,N},
                               z::AbstractArray{<:Real,N},
                               g::AbstractArray{<:Real,N},
                               σ::AbstractArray{<:Real,N},
                               s::AbstractVector{<:AbstractArray{<:Real,N}},
                               src::AbstractVector{<:AbstractString};
                               kwds...) where {T<:AbstractFloat,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, s, src; kwds...)
end

ReducedCalibration{T,N}(obj::ReducedCalibration{T,N}) where {T,N} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T,N}(getfields(obj)...)

#
# Getters.
#
DetectorAxes(obj::ReducedCalibration) = obj.roi
cologlikelihood(obj::ReducedCalibration) = obj.f
detectorbias(obj::ReducedCalibration) = obj.z
detectorgain(obj::ReducedCalibration) = obj.g
detectornoise(obj::ReducedCalibration) = obj.σ
currents(obj::ReducedCalibration) = obj.s
current(obj::ReducedCalibration, k::Integer) = getindex(currents(obj), k)
categories(obj::ReducedCalibration) = obj.src
category(obj::ReducedCalibration, k::Integer) = getindex(categories(obj), k)

#
# Basic operations on ReducedCalibration structure.
#
Base.ndims(obj::ReducedCalibration) = ndims(typeof(obj))
Base.ndims(::Type{<:ReducedCalibration{T,N}}) where {T,N} = N
Base.eltype(obj::ReducedCalibration) = eltype(typeof(obj))
Base.eltype(::Type{<:ReducedCalibration{T}}) where {T} = T
Base.size(obj::ReducedCalibration) = size(DetectorAxes(obj))
Base.size(obj::ReducedCalibration, d::Integer) = size(DetectorAxes(obj), d)
Base.axes(obj::ReducedCalibration) = axes(DetectorAxes(obj))
Base.axes(obj::ReducedCalibration, d::Integer) = axes(DetectorAxes(obj), d)
Base.length(obj::ReducedCalibration) = prod(size(obj))
Base.convert(::Type{T}, obj::ReducedCalibration) where {T<:ReducedCalibration} =
    T(obj)

Base.show(io::IO, obj::ReducedCalibration{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " ReducedCalibration{$T,$N}:")
    for i in 1:length(categories(obj))
        print(io, "\n - src", i, ": \"", identifier(category(obj,i)), "\"")
    end
end

# Allow for `T.(obj)` to work with `T` a floating-point type.
function Broadcast.broadcasted(::Type{T},
                               obj::ReducedCalibration) where {T<:AbstractFloat}
    return ReducedCalibration{T}(obj)
end

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
                            s::AbstractVector{<:AbstractArray},
                            src::AbstractVector{<:Identifiers};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ),
                           _promote_eltype(s)))
    ReducedCalibration{T,N}(roi, f, z, g, σ, s, src; kwds...)
end

function ReducedCalibration{T}(roi::NTuple{N,DetectorAxis},
                               f::AbstractArray,
                               z::AbstractArray,
                               g::AbstractArray,
                               σ::AbstractArray,
                               s::AbstractVector{<:AbstractArray},
                               src::AbstractVector{<:Identifiers};
                               kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, s, src; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray,
                                 z::AbstractArray,
                                 g::AbstractArray,
                                 σ::AbstractArray,
                                 s::AbstractVector{<:AbstractArray},
                                 src::AbstractVector{<:Identifiers};
                                 kwds...) where {T,N}
    T <: AbstractFloat || error("parameter `T` must be a floating-point type")
    length(roi) == N || error("ROI has incompatible number of dimensions")
    length(src) == length(s) || error("incompatible number of categories")
    dims = size(roi)

    function fixarray(A::AbstractArray)
        Base.has_offset_axes(A) && error("array has non-standard indexing")
        eltype(A) <: Real || error("array has incompatible element type")
        ndims(A) == N || error("array has incompatible number of dimensions")
        size(A) == dims ||
            dimension_mismatch("array has incompatible dimensions")
        return convert(Array{T,N}, A)
    end
    ReducedCalibration{T,N}(roi,
                            fixarray(f),
                            fixarray(z),
                            fixarray(g),
                            fixarray(σ),
                            map(fixarray, s),
                            map(identifier, src); kwds...)
end

# Convert pairs like "key1"=>arr1, :key2=>arr2, ... in a list of
# arrays and a list of identifiers.
_getcurrents(args::Pair{<:Union{AbstractString,Symbol},<:AbstractArray}...) =
    (collect(map(x -> x[2], args)),
     collect(map(x -> identifier(x[1]), args)))
_getcurrents() = Int8[], String[]
_getcurrents(s::AbstractVector{<:AbstractArray}, src::AbstractVector) =
    (s, src)

"""
    checkvalues(obj)

throws an error if some values in the reduced calibration object `obj` are
invalid.

"""
function checkvalues(cal::ReducedCalibration)
    checkindices(cal)
    f, z, g, σ, s = cal.f, cal.z, cal.g, cal.σ, cal.s
    for k ∈ eachindex(s)
        all(x -> isfinite(x) && x ≥ 0, s[k]) ||
            error("some invalid values in time-dependent bias")
    end
    all(x -> isfinite(x), z) ||
        error("some invalid values in constant bias")
    all(x -> isfinite(x) && x ≥ 0, g) ||
        error("some invalid values in detector gain")
    all(x -> isfinite(x) && x ≥ 0, σ) ||
        error("some invalid values in readout noise")
end

"""
    checkindices(x)

throws an exception if the fields of object `x` have invalid or incompatible
dimensions or indices.

"""
checkindices(obj::ReducedCalibration) =
    checkindices(typeof(obj), getfields(obj)...)

"""
    checkindices(T, f1, f2, ...)

throws an exception if the fields `f1`, `f2`, ... for an object of type `T`
have invalid or incompatible dimensions or indices.

"""
function checkindices(::Type{<:ReducedCalibration},
                      roi::NTuple{N,DetectorAxis},
                      f::AbstractArray,
                      z::AbstractArray,
                      g::AbstractArray,
                      σ::AbstractArray,
                      s::AbstractVector{<:AbstractArray},
                      src::AbstractVector{<:AbstractString}) where {N}
    for i in 1:N
        length(roi[i]) ≥ 1 || argument_error(
            "invalid detector axis length")
        offset(roi[i]) ≥ 0 || argument_error(
            "invalid detector axis offset")
        binning(roi[i]) ≥ 1|| argument_error(
            "invalid detector axis binning factor")
    end
    inds = axes(roi)
    all(I -> first(I) == 1, inds) || argument_error(
        "ROI has non-standard indices")
    axes(f) == inds || dimension_mismatch("`f` has incompatible indices")
    axes(z) == inds || dimension_mismatch("`z` has incompatible indices")
    axes(g) == inds || dimension_mismatch("`g` has incompatible indices")
    axes(σ) == inds || dimension_mismatch("`σ` has incompatible indices")
    axes(src) == axes(s) || dimension_mismatch(
        "source terms and names have incompatible indices")
    for k ∈ eachindex(s)
        axes(s[k]) == inds || dimension_mismatch(
            "source `s[",k,"]` has incompatible indices")
    end
    return nothing
end
