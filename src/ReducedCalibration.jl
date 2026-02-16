import ..ScientificDetectors: default_valid_pixels_map

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

- Key-value pairs like `"src1" => c1`, `:src2 => c2`, ... of source
  identifiers and arrays corresponding to source terms like the dark current or
  any background flux (in ADU/second).  Arguments `c1`, `c2`, ... are assumed to
  be pixelwise.

- Two arguments: `s = [c1, c2, ...]` and `src = ["src1", "src2", ...]`
  respectively a vector of source terms and of corresponding source
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
    sources(obj)         # yields all sources terms
    sources(obj, k)      # yields k-th source term (in ADU/s)
    sourcesid(obj)       # yields names of sources terms
    sourcesid(obj, k)    # yields name of k-th source term
    nsources(obj)        # # yields number of sources


"""
struct ReducedCalibration{T<:AbstractFloat,N,V<:AbstractArray{Bool,N}}
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::DetectorAxes{N}

    # Co-log-likelihood.
    f::Array{T,N}

    # Zero-level (constant bias in ADU):
    z::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise (in ADU/frame):
    σ::Array{T,N}

    # DIT dependant Standard deviation of the readout noise (in ADU/frame/second):
    σa::Array{T, N}

    # Time dependent source, e.g. dark current and background flux, (in
    # ADU/second), may be empty or zero-filled:
    s::Vector{Array{T,N}}

    # Identifiers of the different sources responsible of the different
    # time-dependent bias terms.
    src::Vector{String}

    # valid pixels map (true = valid pixel)
    vpm::V

    # Inner constructor provided to force using outer constructors.
    function ReducedCalibration{T,N,V}(roi::DetectorAxes{N},
                                       f::AbstractArray{<:Real,N},
                                       z::AbstractArray{<:Real,N},
                                       g::AbstractArray{<:Real,N},
                                       σ::AbstractArray{<:Real,N},
                                       σa::AbstractArray{<:Real,N},
                                       s::AbstractVector{<:AbstractArray{<:Real,N}},
                                       src::AbstractVector{<:AbstractString},
                                       vpm::AbstractArray{Bool,N};
                                       check::Bool = false
                                       ) where {T<:AbstractFloat,N,V<:AbstractArray{Bool,N}}
        checkindices(ReducedCalibration, roi, f, z, g, σ, σa, s, src, vpm)
        obj = new{T,N,V}(roi, f, z, g, σ, σa, s, src, vpm)
        check && checkvalues(obj)
        return obj
    end
end

# Nothing to do for these cases...
ReducedCalibration(obj::ReducedCalibration) = obj
ReducedCalibration{T}(obj::ReducedCalibration{T}) where {T} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{T,N}) where {T,N} = obj
ReducedCalibration{T,N,V}(obj::ReducedCalibration{T,N,V}) where {T,N,V} = obj

function ReducedCalibration(roi::DetectorAxes{N},
                            f::AbstractArray{<:Real,N},
                            z::AbstractArray{<:Real,N},
                            g::AbstractArray{<:Real,N},
                            σ::AbstractArray{<:Real,N},
                            σa::AbstractArray{<:Real,N},
                            s::AbstractVector{<:AbstractArray{<:Real,N}},
                            src::AbstractVector{<:AbstractString},
                            args...;
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ), eltype(σa),
                           map(eltype, s)...))
    ReducedCalibration{T}(roi, f, z, g, σ, σa, s, src, args...; kwds...)
end

for constructor in (:(ReducedCalibration{T}), :(ReducedCalibration{T,N}))
    @eval begin
        $constructor(obj::ReducedCalibration{<:Any,N}) where {T,N} =
            $constructor(getfields(obj)...)

        # Eventually provide type parameter V to call inner constructor.
        function $constructor(roi::DetectorAxes{N},
                              f::AbstractArray{<:Real,N},
                              z::AbstractArray{<:Real,N},
                              g::AbstractArray{<:Real,N},
                              σ::AbstractArray{<:Real,N},
                              σa::AbstractArray{<:Real,N},
                              s::AbstractVector{<:AbstractArray{<:Real,N}},
                              src::AbstractVector{<:AbstractString},
                              vpm::AbstractArray{Bool,N} = default_valid_pixels_map(roi);
                              kwds...) where {T<:AbstractFloat,N}
            ReducedCalibration{T,N,typeof(vpm)}(roi, f, z, g, σ, σa, s, src, vpm; kwds...)
        end
    end
end

ReducedCalibration{T,N,V}(obj::ReducedCalibration{<:Any,N,<:Any}) where {T,N,V} =
    ReducedCalibration{T,N,V}(getfields(obj)...)

#
# Getters.
#
DetectorAxes(obj::ReducedCalibration) = obj.roi
cologlikelihood(obj::ReducedCalibration) = obj.f
detectorbias(obj::ReducedCalibration) = obj.z
detectorgain(obj::ReducedCalibration) = obj.g
detectornoise(obj::ReducedCalibration) = obj.σ
detectornoisedit(obj::ReducedCalibration) = obj.σa
validpixelsmap(obj::ReducedCalibration) = obj.vpm
sources(obj::ReducedCalibration) = obj.s
sources(obj::ReducedCalibration, k::Integer) = getindex(sources(obj), k)
sourcesid(obj::ReducedCalibration) = obj.src
sourcesid(obj::ReducedCalibration, k::Integer) = getindex(sourcesid(obj), k)

nsources(obj::ReducedCalibration) = length(sources(obj))
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
    for i in 1:nsources(obj)
        print(io, "\n - src", i, ": \"", identifier(sourcesid(obj,i)), "\"")
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

# Provide a ROI if not specified and parse source terms.
function ReducedCalibration(f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray, σa::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(map(DetectorAxis, size(f)), f, z, g, σ, σa,
                       _getsources(args...)...; kwds...)
end

function ReducedCalibration{T}(f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray, σa::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(map(DetectorAxis, size(f)), f, z, g, σ, σa,
                          _getsources(args...)...; kwds...)
end

function ReducedCalibration{T,N}(f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray, σa::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(map(DetectorAxis, size(f)), f, z, g, σ, σa,
                            _getsources(args...)...; kwds...)
end

# Parse source terms.
function ReducedCalibration(roi::Tuple{Vararg{DetectorAxis}},
                            f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray, σa::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(roi, f, z, g, σ, σa, _getsources(args...)...; kwds...)
end

function ReducedCalibration{T}(roi::Tuple{Vararg{DetectorAxis}},
                               f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray,
                               σa::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(roi, f, z, g, σ, σa, _getsources(args...)...; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray,
                                 σa::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, σa, _getsources(args...)...; kwds...)
end

function ReducedCalibration(roi::DetectorAxes{N},
                            f::AbstractArray,
                            z::AbstractArray,
                            g::AbstractArray,
                            σ::AbstractArray,
                            σa::AbstractArray,
                            s::AbstractVector{<:AbstractArray},
                            src::AbstractVector{<:Identifiers};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ), eltype(σa),
                           _promote_eltype(s)))
    ReducedCalibration{T,N}(roi, f, z, g, σ,σa, s, src; kwds...)
end

function ReducedCalibration{T}(roi::DetectorAxes{N},
                               f::AbstractArray,
                               z::AbstractArray,
                               g::AbstractArray,
                               σ::AbstractArray,
                               σa::AbstractArray,
                               s::AbstractVector{<:AbstractArray},
                               src::AbstractVector{<:Identifiers};
                               kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, σa, s, src; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray,
                                 z::AbstractArray,
                                 g::AbstractArray,
                                 σ::AbstractArray,
                                 σa::AbstractArray,
                                 s::AbstractVector{<:AbstractArray},
                                 src::AbstractVector{<:Identifiers};
                                 kwds...) where {T,N}
    T <: AbstractFloat || error("parameter `T` must be a floating-point type")
    length(roi) == N || error("ROI has incompatible number of dimensions")
    length(src) == length(s) || error("incompatible number of sources")
    dims = size(roi)

    counter = 0
    function fixarray(A::AbstractArray)
        counter += 1
        Base.has_offset_axes(A) && error(
            "$(nth(counter)) array has non-standard indexing")
        eltype(A) <: Real || error(
            "$(nth(counter)) array has incompatible element type")
        ndims(A) == N || error(
            "$(nth(counter)) array has incompatible number of dimensions")
        size(A) == dims ||
            dimension_mismatch(
                "$(nth(counter)) array has incompatible dimensions")
        return convert(Array{T,N}, A)
    end
    ReducedCalibration{T,N}(roi,
                            fixarray(f),
                            fixarray(z),
                            fixarray(g),
                            fixarray(σ),
                            fixarray(σa),
                            map(fixarray, s),
                            map(identifier, src); kwds...)
end

# Convert pairs like "key1"=>arr1, :key2=>arr2, ... in a list of
# arrays and a list of identifiers.
_getsources(args::Pair{<:Union{AbstractString,Symbol},<:AbstractArray}...) =
    (collect(map(x -> x[2], args)),
     collect(map(x -> identifier(x[1]), args)))
_getsources() = Int8[], String[]
_getsources(s::AbstractVector{<:AbstractArray}, src::AbstractVector) =
    (s, src)

"""
    checkvalues(obj)

throws an error if some values in the reduced calibration object `obj` are
invalid.

"""
function checkvalues(cal::ReducedCalibration)
    checkindices(cal)
    f, z, g, σ, σa, s = cal.f, cal.z, cal.g, cal.σ, cal.σa, cal.s
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
    all(x -> isfinite(x) && x ≥ 0, σa) ||
        error("some invalid values in additional noise")
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
                      roi::DetectorAxes{N},
                      f::AbstractArray,
                      z::AbstractArray,
                      g::AbstractArray,
                      σ::AbstractArray,
                      σa::AbstractArray,
                      s::AbstractVector{<:AbstractArray},
                      src::AbstractVector{<:AbstractString},
                      vpm::AbstractArray) where {N}
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
    axes(σa) == inds || dimension_mismatch("`σa` has incompatible indices")
    axes(vpm) == inds || dimension_mismatch("`vpm` has incompatible indices")
    axes(src) == axes(s) || dimension_mismatch(
        "source terms and names have incompatible indices")
    for k ∈ eachindex(s)
        axes(s[k]) == inds || dimension_mismatch(
            "source `s[",k,"]` has incompatible indices")
    end
    return nothing
end
