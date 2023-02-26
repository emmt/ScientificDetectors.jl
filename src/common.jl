"""
    ScientificDetectors.OnlineStatistics{T,N}

is an alias for:

    MultivariateOnlineStatistics.IndependentStatistics{2,T,N,Array{T,N}}

"""
const OnlineStatistics{T,N} = IndependentStatistics{2,T,N,Array{T,N}}

"""
    A = DetectorAxis(len; off=0, bin=1, step=1)

builds an instance `A` of `DetectorAxis` describing a detector axis with `len`
samples starting at offset `off` with respect to the corresponding sensor egde
and with a binning factor `bin`, and sampling `step`. The offset `off`, the
binning factor `bin`, and the sampling `step` are in units of sensor samples
(e.g., *pixels*), the length `len` is in units of macro-samples (i.e., `bin`
sensor samples each).

Basic methods (`A` is an instance of `DetectorAxis`, `ROI` is a tuple of
`DetectorAxis`):

    length(A)    # the length `len`
    offset(A)    # the offset `off`
    binning(A)   # the binning factor `bin`
    step(A)      # the sampling step in pixels `step`
    size(ROI)    # the size of ROI
    size(ROI, i) # the length of the i-th dimension of ROI

Other methods:

    get(DetectorAxis, i, src)      # the i-th detector axis of `src`
    get(Vector{DetectorAxis}, src) # all detector axes of `src`
    merge!(dst, ROI)               # set detector axes of `dst`

here the source `src` and the destination `dst` can be instances of
`FitsHeader` or of `FitsHDU`, `ROI` is a vector or a tuple of `DetectorAxis`.

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
    stp::Int # step (in pixels)
end

const DetectorAxes{N} = NTuple{N,DetectorAxis}

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

Base.size(roi::DetectorAxes) = map(length, roi)
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
    return map(DetectorAxis, size(A))
end

DetectorAxes(I::DetectorAxisTypes...) = DetectorAxes(I)
DetectorAxes(I::Tuple{Vararg{DetectorAxisTypes}}) = map(DetectorAxis, I)

function Base.merge!(dst::FitsHeader,
                     prm::Union{Tuple{Vararg{DetectorAxis}},
                                AbstractVector{<:DetectorAxis}})
    _merge_axes!(dst, prm)
end

function Base.merge!(dst::FitsHDU,
                     prm::Union{Tuple{Vararg{DetectorAxis}},
                                AbstractVector{<:DetectorAxis}})
    _merge_axes!(dst, prm)
end

function _merge_axes!(dst::Union{FitsHeader,FitsHDU},
                      prm::Union{Tuple{Vararg{DetectorAxis}},
                                AbstractVector{<:DetectorAxis}})
    n = length(prm)
    for i in 1:n
        dst["NAXIS$i"] = (prm[i].len, "length of data axis $i")
    end
    for i in 1:n
        dst["OFF$i"] = (prm[i].off, "offset along axis $i")
    end
    for i in 1:n
        dst["BIN$i"] = (prm[i].bin, "binning factor of axis $i")
    end
    for i in 1:n
        dst["STP$i"] = (prm[i].bin, "sampling step axis $i")
    end
    return dst
end

function Base.get(::Type{DetectorAxis}, i::Integer,
                  src::Union{FitsHeader,FitsHDU})
    i ≥ 1 || throw(ArgumentError("invalid axis number $i"))
    len = src["NAXIS$i"].integer
    off = getvalue(Int, src, "OFF$i", 0)
    bin = getvalue(Int, src, "BIN$i", 1)
    stp = getvalue(Int, src, "STP$i", 1)
    return DetectorAxis(len, off, bin, stp)
end

function Base.get(::Type{Vector{DetectorAxis}},
                  src::Union{FitsHeader,FitsHDU})
    n = src["NAXIS"].integer
    res = Vector{DetectorAxis}(undef, n)
    for i in 1:n
        res[i] = get(DetectorAxis, i, src)
    end
    return res
end

function Base.get(::Type{DetectorAxes{N}},
                  src::Union{FitsHeader,FitsHDU}) where {N}
    return ntuple(i -> get(DetectorAxis, i, src), Val(N))
end

#------------------------------------------------------------------------------
# FITS CARD VALUES

# FIXME: The following methods should be provided by EasyFITS.

"""
    getvalue([T,] H, key)

yields the value of FITS keyword `key` in `H` throwing an error if `key` does
not exist. If `T` is specified, the keyword value is converted to type `T`.

"""
getvalue(H::Union{FitsHDU,FitsHeader}, key::AbstractString) = H[key].value()
getvalue(T::Type, H::Union{FitsHDU,FitsHeader}, key::AbstractString) = H[key].value(T)

"""
    getvalue([T,] H, key, def)

yields the value of FITS keyword `key` in `H` or `def` if `key` does not exist.
If `T` is specified and keyword `key` exists, the keyword value is converted to
type `T`.

"""
getvalue(H::Union{FitsHDU,FitsHeader}, key::AbstractString, def) =
    (card = get(H, key, nothing)) === nothing ? def : card.value()
getvalue(T::Type, H::Union{FitsHDU,FitsHeader}, key::AbstractString, def) =
    (card = get(H, key, nothing)) === nothing ? def : card.value(T)

"""
    matchvalue(val, card)
    matchvalue(card, val)

yield whether `val` is equal to the value of the FITS card `card`.

"""
matchvalue(val, card::FitsCard) = matchvalue(card, val)
matchvalue(card::FitsCard, val::EasyFITS.Undefined) = card.type == FITS_UNDEFINED
matchvalue(card::FitsCard, val::Nothing) = card.type == FITS_COMMENT
matchvalue(card::FitsCard, val::AbstractString) =
    card.type == FITS_STRING ? card.string == val : false
matchvalue(card::FitsCard, val::Number) =
    card.type == FITS_LOGICAL ? card.logical == val :
    card.type == FITS_INTEGER ? card.integer == val :
    card.type == FITS_FLOAT   ? card.float   == val :
    card.type == FITS_COMPLEX ? card.complex == val : false
matchvalue(card::FitsCard, val) = false

"""
    matchvalue(H, key, val)

yields whether `H` has a FITS keyword `key` whose value is equal to `val`.

"""
matchvalue(H::Union{FitsHDU,FitsHeader}, key::AbstractString, val) =
    (card = get(H, key, nothing)) === nothing ? false : matchvalue(card, val)

"""
    f = KeywordMatcher(key, val)

yields a callable object `f` which can be called as:

    f(H)

to yield whether `H` has a FITS keyword `key` whose value is equal to `val`.

"""
struct KeywordMatcher{V} <: Function
    key::String
    value::V
end
(obj::KeywordMatcher)(H::Union{FitsHDU,FitsHeader}) =
    matchvalue(H, obj.key, obj.val)

#------------------------------------------------------------------------------
# IDENTIFIERS

"""
    ScientificDetectors.Identifiers

is the union of types acceptable for identifiers (strings, symbols, or
integers).

"""
const Identifiers = Union{AbstractString,Symbol,Integer}

"""
    identifier(key) -> str

converts `key` into a string identifier.  Argument `key` can be of any type
part of the union `Identifiers` (a string, a symbol or an integer).  Also works
if argument is a tuple or an array of identifiers.

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)
identifier(A::AbstractArray{String}) = A
identifier(A::AbstractArray{<:Identifiers}) = map(identifier, A)
identifier(A::Tuple{Vararg{String}}) = A
identifier(A::Tuple{Vararg{<:Identifiers}}) = map(identifier, A)

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
# NOTE: `string(n)*ordinal_suffix(n)` is about 3 times faster (76.5ns) than
#       `string(n,ordinal_suffix(n))` or `"$n$(ordinal_suffix(n))"` which are
#       equally slow (208ns).

"""
    ordinal_suffix(n)

yields the ordinal suffix `"st"`, `"nd"`, `"rd"`, or `"th"` corresponding
to the value of the integer `n`.

"""
ordinal_suffix(n::Integer) =
    (d = abs(n)%10) == 1 ? "st" : d == 2 ? "nd" : d == 3 ? "rd" : "th"

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
