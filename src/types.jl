struct DetectorAxis
    len::Int # number of macro-pixels
    off::Int # offset (in pixels) of the ROI relative to detector edge
    bin::Int # binning factor (in pixels)
    stp::Int # step (in pixels)
end

# Union of types that can be interpreted as a `DetectorAxis`.
const DetectorAxisLike = Union{<:Integer, DetectorAxis,
                               <:OrdinalRange{<:Integer,<:Integer}}

"""
Instances of type `DetectorAxes{N}` (note the plural) store a `N`-tuple of `DetectorAxis`
instances.

Having `DetectorAxes{N}` a distinctive type rather than an alias for
`NTuple{N,DetectorAxis}` is to avoid type piracy, notably for the empty tuple when `N = 0`.

"""
struct DetectorAxes{N}
    axes::NTuple{N,DetectorAxis}
    DetectorAxes(axes::NTuple{N,DetectorAxis}) where {N} = new{N}(axes)
end
