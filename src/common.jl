"""
    R = DetectorAxis(len; off=0, bin=1, step=1)

Return an object describing a detector axis with `len` (macro-)pixels. Along this axis, the
(macro-)pixels start at offset `off` from the sensor edge, have a binning factor `bin`, and
sampling `step`. Parameters `off`, `bin`, and `step` are in units of physical pixels.

The layout is illustrated below for detector axis parameters `off = 4`, `bin = 2`, and `step
= 3`. `[ ]` denote unused physical pixels while `[i]` denote physical pixels that are part
of the `i`-th macro-pixel:

```
 [ ] [ ] [ ] [ ] [1] [1] [ ] [2] [2] [ ] [3...
|<─────────────>|<─────>|   |<─────>|   |
       off      |  bin  |   |  bin  |   |
                |<─────────>|<─────────>|
                     step        step
```

Basic methods (`R` is an instance of `DetectorAxis`, `ROI` is an instance of
`DetectorAxes`):

    length(R)    # the length `len`
    offset(R)    # the offset `off`
    binning(R)   # the binning factor `bin`
    step(R)      # the sampling step in pixels `step`
    size(ROI)    # the size of ROI
    size(ROI, i) # the length of the i-th dimension of ROI

Other methods:

    DetectorAxis(A, i)      # the i-th detector axis of `A`

"""
struct DetectorAxis
    # Along the considered dimension:
    len::Int # number of macro-pixels
    off::Int # offset (in pixels) of the ROI relative to detector edge
    bin::Int # binning factor (in pixels)
    stp::Int # step (in pixels)
end

"""
Instances of type `DetectorAxes{N}` (note the plural) store a `N`-tuple of `DetectorAxis`
instances.

Having a dedicated package type rather than extending `NTuple{N,DetectorAxis}` directly
keeps basic methods such as `size`, `axes`, and conversions attached to a type owned by
`ScientificDetectors` and avoids accidental method piracy on tuples, notably for the empty
tuple when `N = 0`.

"""
struct DetectorAxes{N}
    data::NTuple{N,DetectorAxis}
end

"""
    roi = DetectorAxes(B)
    roi = DetectorAxes{N}(B)

Build an object representing by a `N`-tuple of `DetectorAxis` the region of interest (ROI)
geometry of the detector for object `B`. Optional parameter `N` is the number of dimensions
of the detector.

Call `Tuple(roi)` to retrieve the `N`-tuple of `DetectoAxis` instances.

Call the `range` function as follows to retrieve the indices along `k`-th dimension of array
`A` of the first physical pixels for the macro-pixels defined by for detector axes `R`:

    range(A, R, k)


    get(Vector{DetectorAxis}, src) # all detector axes of `src`
    merge!(dst, ROI)               # set detector axes of `dst`

"""
# Union of types that can be interpreted as DetectorAxis.
const DetectorAxisTypes = Union{<:Integer,DetectorAxis,
                                <:OrdinalRange{<:Integer,<:Integer}}

DetectorAxis(A::DetectorAxis) = A

DetectorAxis(len::Integer; off::Integer=0, bin::Integer=1, step::Integer=1) =
    DetectorAxis(len, off, bin, step)

DetectorAxis(R::OrdinalRange{<:Integer,<:Integer}) = begin
    step(R) > 0 || argument_error("step must be positive")
    DetectorAxis(length(R), offset(R), binning(R), step(R))
end

function Base.range(A::AbstractArray, R::DetectorAxis, k::Integer)
    origin = as(Int, first(axes(A, k))) + offset(R)
    return origin : step(R) : origin + step(R)*(length(R) - 1)
end

function Base.AbstractUnitRange(r::AbstractUnitRange{<:Integer}, x::DetectorAxis)
    # NOTE: yields the index in physical array representing the detector of the first
    # physical pixel in each macro-pixel along a dimension.
    start = first(r) + offset(x)
    stop = start + step(x)*(length(x) - 1)
    return start:step(x):stop
end

# Accessors.
Base.length(A::DetectorAxis) = getfield(A, :len)
offset(A::DetectorAxis) = getfield(A, :off)
binning(A::DetectorAxis) = getfield(A, :bin)
Base.step(A::DetectorAxis) = getfield(A, :stp)

# Extend methods for ranges.
offset(R::OrdinalRange{<:Integer,<:Integer}) = Int(first(R)) - 1
binning(R::OrdinalRange{<:Integer,<:Integer}) = 1

@doc @doc(DetectorAxis) offset
@doc @doc(DetectorAxis) binning

# Make an instance of `DetectorAxes` behave as a n-tuple.
Base.eltype(::Type{<:DetectorAxes}) = DetectorAxis
Base.length(::DetectorAxes{N}) where {N} = N
@propagate_inbounds Base.getindex(roi::DetectorAxes, i::Integer) = getindex(Tuple(roi), i)
Base.iterate(roi::DetectorAxes, i::Integer = 1) =
    (1 ≤ i ≤ length(roi) ? (roi[i], i + 1) : nothing)
Base.Tuple(roi::DetectorAxes) = getfield(roi, :data)

Base.size(roi::DetectorAxes) = map(length, Tuple(roi))
Base.size(roi::DetectorAxes{N}, i::Integer) where {N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? length(@inbounds roi[i]) : 1)
Base.axes(roi::DetectorAxes) = map(Base.OneTo, size(roi))
Base.axes(roi::DetectorAxes, i::Integer) = Base.OneTo(size(roi, i))

Base.show(io::IO, A::DetectorAxis) =
    print(io, "DetectorAxis(", length(A), "; off=", offset(A),
          ", bin=", binning(A), ", step=", step(A), ")")

DetectorAxes{N}(A::AbstractArray{<:Any,N}) where {N} = DetectorAxes(A)
DetectorAxes{N}(I::DetectorAxisTypes...) where {N} = DetectorAxes{N}(I)
DetectorAxes{N}(I::NTuple{N,DetectorAxisTypes}) where {N} = DetectorAxes(I)

DetectorAxes(A::AbstractArray) = begin
    Base.has_offset_axes(A) && error("array has non-standard indexing")
    return DetectorAxes(map(DetectorAxis, size(A)))
end

DetectorAxes(I::DetectorAxisTypes...) = DetectorAxes(I)
DetectorAxes(I::NTuple{N,DetectorAxisTypes}) where {N} =
    DetectorAxes{N}(map(DetectorAxis, I))

DetectorAxis(A::DetectorAxes, i::Integer) = A[i]

default_valid_pixels_map(roi::DetectorAxes) = FastUniformArray(true, size(roi))

#------------------------------------------------------------------------------
# IDENTIFIERS

"""
    ScientificDetectors.Identifiers

Union of types acceptable for identifiers (strings, symbols, or integers).

"""
const Identifiers = Union{AbstractString,Symbol,Integer}

"""
    identifier(key) -> str

Converts `key` into a string identifier. Argument `key` can be of any type in union
`Identifiers` (a string, a symbol or an integer). Argument can be a tuple or an array of
identifiers.

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)
identifier(A::AbstractArray{String}) = A
identifier(A::AbstractArray{<:Identifiers}) = map(identifier, A)
identifier(A::Tuple{Vararg{String}}) = A
identifier(A::Tuple{Vararg{Identifiers}}) = map(identifier, A)

#------------------------------------------------------------------------------
#
# Sampler to provide samples given an array.
#
struct Sampler{T,N,Np1,A<:AbstractArray{T,Np1}}
    data::A
    inds::NTuple{N,Colon}
    function Sampler{T,N,Np1,A}(data::A) where {T,N,Np1,A<:AbstractArray{T,Np1}}
        Np1 == N + 1 || error("Np1 ≠ N + 1")
        Np1 ≥ 2 || error("insufficient number of dimensions")
        Base.has_offset_axes(data) && error(
            "data array has non-standard indexing")
        samples = size(data, Np1)
        samples ≥ 2 || error("insufficient number of samples")
        new{T,N,Np1,A}(data, colons(N))
    end
end

Sampler(data::A) where {T,N,A<:AbstractArray{T,N}} = Sampler{T,N-1,N,A}(data)

StatsBase.nobs(A::Sampler{T,N,Np1}) where {T,N,Np1} = size(A.data, Np1)

Base.eltype(A::Sampler{T,N}) where {T,N} = T
Base.ndims(A::Sampler{T,N}) where {T,N} = N
Base.length(A::Sampler) = nobs(A)
Base.size(A::Sampler) = size(A.data)[1:end-1]
Base.size(A::Sampler{T,N}, i::Integer) where {T,N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? size(A.data, i) : 1)

Base.show(io::IO, obj::Sampler{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " Sampler{$T,$N}: samples = ", nobs(obj))
end

Base.iterate(A::Sampler, i = 1) =
    (1 ≤ i ≤ nobs(A) ? (view(A.data, A.inds..., i), i+1) : nothing)

"""
    Ticker(start, stop) -> obj

builds a callable object which yields `start`, `start + step`, `start +
2*step`, ... each time it is called.

"""
mutable struct Ticker{T<:Number} <: Function
    value::T
    start::T
    step::T
end
Ticker(start::Number, step::Number) = Ticker(promote(start, step)...)
Ticker(start::T, step::T) where {T<:Number} = Ticker{T}(start, start, step)
function (obj::Ticker)()
    value = obj.value
    obj.value = value + obj.step
    return value
end
function Base.Iterators.reset!(obj::Ticker)
    obj.value = obj.start
    return obj
end
Base.take!(obj::Ticker) = obj()

"""
    exposuretime(obj) -> Δt

yields the exposure time of object `obj`.

See also [`PreprocessingParameters`](@ref).

"""
function exposuretime end

"""
    getfields(x) -> (getfield(x,1), getfield(x,2), ..., getfield(x,nfields(x)))

yields a tuple of the fields of object `x`.

"""
getfields(x) = ntuple(i -> getfield(x, i), Val(nfields(x)))

"""
    nth(n)

yields the string `"\$n\$(ordinal_suffix(n))"`.

"""
nth(n::Integer) = string(n)*ordinal_suffix(n)
# NOTE: `string(n)*ordinal_suffix(n)` is about 3 times faster than
#       `string(n,ordinal_suffix(n))` or `"$n$(ordinal_suffix(n))"` which are
#       equally slow.

"""
    ordinal_suffix(n)

yields the ordinal suffix `"st"`, `"nd"`, `"rd"`, or `"th"` corresponding
to the value of the integer `n`.

"""
function ordinal_suffix(n::Integer)
    if n > zero(n)
        ten = oftype(n, 10)
        if rem(div(n, ten), ten) != one(n)
            # Number is positive and the tens digit is not 1.
            r = rem(n, ten)
            if r == oftype(r, 1)
                return "st"
            elseif r == oftype(r, 2)
                return "nd"
            elseif r == oftype(r, 3)
                return "rd"
            end
        end
    end
    return "th"
end

@noinline argument_error(args...) =
    argument_error(string(args...))
@noinline argument_error(mesg::AbstractString) =
    throw(ArgumentError(mesg))

@noinline dimension_mismatch() =
    dimension_mismatch("arguments have incompatible dimensions")
@noinline dimension_mismatch(args...) =
    dimension_mismatch(string(args...))
@noinline dimension_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))
