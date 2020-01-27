"""

```julia
DetectorAxis(len; off=0, bin=1) -> obj
```

yields an instance of `DetectorAxis` describing a detector axis with `len`
samples starting at offset `off` with respect to the corresponding sensor egde
and with a binning factor `bin`.  The offset `off` and the binning factor `bin`
are in units sensor samples (e.g. *pixels*), the length `len` is in units of
macro-samples (i.e. `bin` sensor samples each).

Basic methods:

```julia
length(obj)    # yields the length `len`
offset(obj)    # yields the offset `off`
binning(obj)   # yields the binning factor `bin`
size(roi)      # yields the size of roi, a N-tuple of DetectorAxis
size(roi, i)   # yields the length of the i-th dimension of roi
```

Other methods:

```julia
get(DetectorAxis, i, src)      # yields the i-th detector axis of `src`
get(Vector{DetectorAxis}, src) # yields a vector of detector axes of source `src`
merge!(dst, obj)               # set detector axes of`dst`
```

here the source `src` and the destination `dst` can be instances of
`FitsHeader` or `FitsImage`, `obj` is a vector or a tuple of `DetectorAxis`.

"""
struct DetectorAxis
    len::Int # Number of (macro-)samples along the dimensions.
    off::Int # Offset of the ROI with respect to the sensor (in physical samples).
    bin::Int # Binning factor (in physical samples).
end
DetectorAxis(len::Integer; off::Integer=0, bin::Integer=1) =
    DetectorAxis(len, off, bin)

offset(obj::DetectorAxis) = obj.off
binning(obj::DetectorAxis) = obj.bin

@doc @doc(DetectorAxis) offset
@doc @doc(DetectorAxis) binning

Base.length(obj::DetectorAxis) = obj.len
Base.size(roi::NTuple{N,DetectorAxis}) where {N} = map(x -> length(x), roi)
Base.size(roi::NTuple{N,DetectorAxis}, i::Integer) where {N} =
    (i < 1 ? error("out of range dimension index") :
     i ≤ N ? length(roi[i]) : 1)

Base.show(io::IO, obj::DetectorAxis) =
    print(io, "DetectorAxis(", length(obj), "; off=", offset(obj),
          ", bin=", binning(obj), ")")

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

function Base.get(::Type{NTuple{N,DetectorAxis}},
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
        @assert Np1 == N + 1
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

```julia
regionofinterest(obj) -> roi
```

yields the region of interest (ROI) of object `obj`.

See also [`DetectorAxis`](@ref).

"""
function regionofinterest end

"""

```julia
exposuretime(obj) -> Δt
```

yields the exposure time of object `obj`.

See also [`PreprocessingParameters`](@ref).

"""
function exposuretime end

"""

```julia
numberofsamples(obj)
```

yields the number of samples used for statistics stored in object `obj`.

See also [`exposuretime`](@ref), [`SampleStatistics`](@ref).

"""
function numberofsamples end

@noinline dimension_mismatch() =
    dimension_mismatch("arguments have incompatible dimensions")

@noinline dimension_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))
