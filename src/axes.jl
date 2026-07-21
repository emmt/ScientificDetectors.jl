"""
    R = DetectorAxis(len; off=0, bin=1, step=bin)

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


    range(A::AbstractArray, R::DetectorAxis, k::Integer) -> rng

return the indices along `k`-th dimension of array `A` of the first physical
pixels for the macro-pixels defined by for detector axes `R`:


    DetectorAxis(obj, i) # i-th detector axis of `obj`

"""
function DetectorAxis(len::Integer; off::Integer=0, bin::Integer=1, step::Integer=bin)
    return DetectorAxis(len, off, bin, step)
end

function DetectorAxis(R::AbstractRange{<:Integer})
    step(R) > 0 || argument_error("step must be positive")
    return DetectorAxis(length(R); off=offset(R), bin=binning(R), step=step(R))
end

DetectorAxis(A::DetectorAxis) = A

# For each type of the `DetectorAxisLike` union, conversion to `DetectorAxis` must be implemented.
Base.convert(::Type{DetectorAxis}, x::DetectorAxis) = x
Base.convert(::Type{DetectorAxis}, x::DetectorAxisLike) = DetectorAxis(x)::DetectorAxis

DetectorAxis(A::DetectorAxes, i::Integer) = A[i]

function Base.range(A::AbstractArray, R::DetectorAxis, k::Integer)
    inds = axes(A, k)
    start = Int(first(inds))::Int + offset(R)
    stop = start + step(R)*(length(R) - 1)
    # start ≥ first(inds) || error("out of range first index")
    # stop ≤ last(inds) || error("out of range last index")
    return start : step(R) : stop
end

function Base.AbstractUnitRange(r::AbstractUnitRange{<:Integer}, x::DetectorAxis)
    # NOTE: yields the index in physical array representing the detector of the first
    # physical pixel in each macro-pixel along a dimension.
    start = first(r) + offset(x)
    stop = start + step(x)*(length(x) - 1)
    return start:step(x):stop
end

# Basic accessors.
Base.length(A::DetectorAxis) = getfield(A, :len)
Base.step(A::DetectorAxis) = getfield(A, :stp)

"""
    ScientificDetectors.offset(r::Union{DetectorAxis,AbstractRange{<:Integer}})

Return the offset of `r`.

"""
offset(A::DetectorAxis) = getfield(A, :off)
offset(R::AbstractRange{<:Integer}) = Int(first(R)) - 1

"""
    ScientificDetectors.binning(r::Union{DetectorAxis,AbstractRange{<:Integer}})

Return the binning factor of `r`.

"""
binning(A::DetectorAxis) = getfield(A, :bin)
binning(R::AbstractRange{<:Integer}) = 1

Base.show(io::IO, A::DetectorAxis) = print(
    io, "DetectorAxis(", length(A), "; off=", offset(A),
    ", bin=", binning(A), ", step=", step(A), ")")

"""
    roi = DetectorAxes(B)
    roi = DetectorAxes{N}(B)

Build an object representing by a `N`-tuple of [`DetectorAxis`](@ref) the region of interest
(ROI) geometry of the detector for object `B`. Optional parameter `N` is the number of
dimensions of the detector.

Call `Tuple(roi)` to retrieve the `N`-tuple of [`DetectorAxis`](@ref) instances.

"""
DetectorAxes{N}(A::AbstractArray{<:Any,N}) where {N} = DetectorAxes(A)
function DetectorAxes(A::AbstractArray)
    Base.has_offset_axes(A) && argument_error("array has non-standard indexing")
    return DetectorAxes(map(DetectorAxis, size(A)))
end

DetectorAxes{N}(I::DetectorAxisLike...) where {N} = DetectorAxes{N}(I)
DetectorAxes{N}(I::NTuple{N,DetectorAxisLike}) where {N} = DetectorAxes(I)
DetectorAxes(I::DetectorAxisLike...) = DetectorAxes(I)
function DetectorAxes(I::NTuple{N,DetectorAxisLike}) where {N}
    return DetectorAxes(map(DetectorAxis, I))
end

DetectorAxes{N}(A::DetectorAxes{N}) where {N} = A
DetectorAxes(A::DetectorAxes) = A
Base.convert(::Type{T}, A::T) where {T<:DetectorAxes} = A
Base.convert(::Type{T}, A::Any) where {T<:DetectorAxes} = T(A)::T

# Accessor.
Base.Tuple(roi::DetectorAxes) = getfield(roi, :axes)

# Make an instance of `DetectorAxes` behave as a n-tuple.
Base.eltype(::Type{<:DetectorAxes}) = DetectorAxis
Base.length(::DetectorAxes{N}) where {N} = N
@propagate_inbounds Base.getindex(roi::DetectorAxes, i::Integer) = getindex(Tuple(roi), i)
Base.iterate(roi::DetectorAxes, i::Integer = 1) =
    (1 ≤ i ≤ length(roi) ? (roi[i], i + 1) : nothing)

Base.size(roi::DetectorAxes) = map(length, Tuple(roi))
Base.size(roi::DetectorAxes{N}, i::Integer) where {N} =
    i < 1 ? argument_error("out of range dimension index") :
    i > N ? 1 : length(@inbounds roi[i])

Base.axes(roi::DetectorAxes) = map(Base.OneTo, size(roi))
Base.axes(roi::DetectorAxes, i::Integer) = Base.OneTo(size(roi, i))
