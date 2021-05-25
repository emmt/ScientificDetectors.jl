"""
    ScientificDetectors.OnlineStatistics{T,N}

is an alias for:

    MultivariateOnlineStatistics.IndependentStatistics{2,T,N,Array{T,N}}

"""
const OnlineStatistics{T,N} = IndependentStatistics{2,T,N,Array{T,N}}

"""
    A = DetectorAxis(len; off=0, bin=1)

builds an instance `A` of `DetectorAxis` describing a detector axis with `len`
samples starting at offset `off` with respect to the corresponding sensor egde
and with a binning factor `bin`.  The offset `off` and the binning factor `bin`
are in units sensor samples (e.g., *pixels*), the length `len` is in units of
macro-samples (i.e., `bin` sensor samples each).

Basic methods (`A` is an instance of `DetectorAxis`, `ROI` is a tuple of
`DetectorAxis`):

    length(A)    # the length `len`
    offset(A)    # the offset `off`
    binning(A)   # the binning factor `bin`
    size(ROI)    # the size of ROI
    size(ROI, i) # the length of the i-th dimension of ROI

Other methods:

    get(DetectorAxis, i, src)      # the i-th detector axis of `src`
    get(Vector{DetectorAxis}, src) # all detector axes of `src`
    merge!(dst, ROI)               # set detector axes of `dst`

here the source `src` and the destination `dst` can be instances of
`FitsHeader` or `FitsImage`, `ROI` is a vector or a tuple of `DetectorAxis`.

    DetectorAxes(B)

yields the detector geometry settings for object `B` (note the plural) as an
`N`-tuple of `DetectorAxis`, `N` being the number of dimensions of the
detector.

"""
struct DetectorAxis
    # Along the considered dimension:
    len::Int # number of macro-pixels
    off::Int # offset (in pixels) of the ROI relative to detector edge
    bin::Int # binning factor (in pixels)
end

const DetectorAxes{N} = NTuple{N,DetectorAxis}

# Union of types that can be interpreted as DetectorAxis.
const DetectorAxisTypes = Union{<:Integer,DetectorAxis,
                                <:OrdinalRange{<:Integer,<:Integer}}

DetectorAxis(A::DetectorAxis) = A

DetectorAxis(len::Integer; off::Integer=0, bin::Integer=1) =
    DetectorAxis(len, off, bin)

DetectorAxis(R::OrdinalRange{<:Integer,<:Integer}) = begin
    step(R) > 0 || argument_error("step must be positive")
    DetectorAxis(length(R), offset(R), binning(R))
end

offset(A::DetectorAxis) = A.off
offset(R::OrdinalRange{<:Integer,<:Integer}) = Int(first(R)) - 1
binning(A::DetectorAxis) = A.bin
binning(R::OrdinalRange{<:Integer,<:Integer}) = Int(step(R))

@doc @doc(DetectorAxis) offset
@doc @doc(DetectorAxis) binning

Base.length(A::DetectorAxis) = A.len
Base.size(roi::DetectorAxes{N}) where {N} = map(length, roi)
Base.size(roi::DetectorAxes{N}, i::Integer) where {N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? length(roi[i]) : 1)

Base.show(io::IO, A::DetectorAxis) =
    print(io, "DetectorAxis(", length(A), "; off=", offset(A),
          ", bin=", binning(A), ")")

DetectorAxes{N}(A::AbstractArray{<:Any,N}) where {N} = DetectorAxes(A)
DetectorAxes{N}(I::DetectorAxisTypes...) where {N} = DetectorAxes{N}(I)
DetectorAxes{N}(I::NTuple{N,DetectorAxisTypes}) where {N} = DetectorAxes(I)

DetectorAxes(A::AbstractArray) = begin
    Base.has_offset_axes(A) && error("array has non-standard indexing")
    map(DetectorAxis, size(A))
end

DetectorAxes(I::DetectorAxisTypes...) = DetectorAxes(I)
DetectorAxes(I::Tuple{Vararg{DetectorAxisTypes}}) = map(DetectorAxis, I)

function Base.merge!(dst::FitsHeader,
                     prm::Union{Tuple{Vararg{T}},AbstractVector{T}}
                     ) where {T<:DetectorAxis}
    n = length(prm)
    for d in 1:n
        sfx = string(d)
        dst["NAXIS"*sfx] = (prm[d].len, "length of data axis "*sfx)
    end
    for d in 1:n
        sfx = string(d)
        dst["OFF"*sfx] = (prm[d].off, "offset along axis "*sfx)
    end
    for d in 1:n
        sfx = string(d)
        dst["BIN"*sfx] = (prm[d].bin, "binning factor of axis "*sfx)
    end
    return dst
end

function Base.get(::Type{DetectorAxis}, i::Integer,
                  src::Union{FitsHeader,FitsImage})
    @assert 1 ≤ i
    sfx = string(i)
    len = src["NAXIS"*sfx]
    off = get(src, "OFF"*sfx, 0)
    bin = get(src, "BIN"*sfx, 1)
    return DetectorAxis(len, off, bin)
end

function Base.get(::Type{Vector{DetectorAxis}},
                  src::Union{FitsHeader,FitsImage})
    n = src["NAXIS"]
    res = Vector{DetectorAxis}(undef, n)
    for i in 1:n
        res[i] = get(DetectorAxis, i, src)
    end
    return res
end

function Base.get(::Type{DetectorAxes{N}},
                  src::Union{FitsHeader,FitsImage}) where {N}
    ntuple(i -> get(DetectorAxis, i, src), Val(N))
end

#
# Sampler to provide samples given an array.
#
struct Sampler{T,N,Np1,A<:AbstractArray{T,Np1}}
    data::A
    inds::NTuple{N,Colon}
    function Sampler{T,N,Np1,A}(data::A) where {T,N,Np1,A<:AbstractArray{T,Np1}}
        Np1 == N + 1 || error("Np1 ≠ N + 1")
        Np1 ≥ 2 || error("insufficient number of dimensions")
        Base.has_offset_axes(data) && error("data array has non-standard indexing")
        samples = size(data, Np1)
        samples ≥ 2 || error("insufficient number of samples")
        new{T,N,Np1,A}(data, colons(N))
    end
end

Sampler(data::A) where {T,N,A<:AbstractArray{T,N}} = Sampler{T,N-1,N,A}(data)

numberofsamples(A::Sampler{T,N,Np1}) where {T,N,Np1} = size(A.data, Np1)

Base.eltype(A::Sampler{T,N}) where {T,N} = T
Base.ndims(A::Sampler{T,N}) where {T,N} = N
Base.length(A::Sampler) = numberofsamples(A)
Base.size(A::Sampler) = size(A.data)[1:end-1]
Base.size(A::Sampler{T,N}, i::Integer) where {T,N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? size(A.data, i) : 1)

Base.show(io::IO, obj::Sampler{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " Sampler{$T,$N}: samples = ", numberofsamples(obj))
end

Base.iterate(A::Sampler, i = 1) =
    (1 ≤ i ≤ numberofsamples(A) ? (view(A.data, A.inds..., i), i+1) : nothing)

"""
    exposuretime(obj) -> Δt

yields the exposure time of object `obj`.

See also [`PreprocessingParameters`](@ref).

"""
function exposuretime end

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
