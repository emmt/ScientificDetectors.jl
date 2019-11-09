module Preprocessing

import ..ScientificDetectors: ReducedCalibration, process, process!

using ArrayTools

"""

`ScientificDetectors.PreprocessingParameters{T}` stores the pre-processing
parameters with `T` the floating-point type for the computations.

Constructor is called as:

```julia
ScientificDetectors.PreprocessingParameters([T,] a, b, u, v) -> obj
```

where `T` is the floating-point type for the computations (optional, if
unspecified, deduced from the element type of the other arguments), `a` is the
amplitude correction factor (in flux units per ADU), `b` is the bias correction
(in ADU), `u` and `v` are variance terms.  Arguments `a`, `b`, `u` and `v` are
pixelwise, they are broadcast to common dimensions (which should be that of the
detector) and their elements converted to the same type `T`.


It is also possible to convert calibration parameters to preprocessing
parameters:

```julia
ScientificDetectors.PreprocessingParameters([T,] cal::ReducedCalibration, Δt=0) -> obj
```

with `Δt` the exposure time in seconds.


The pre-processing of pixel `raw[i]` of an image given by the detector leads to
compute the calibrated pixel value `dat[i]` and its corresponding precision
`wgt[i]` as follows:

```julia
dat[i] = (raw[i] - b[i])*a[i]
wgt[i] = u[i]/(max(zero(T), dat[i]) + v[i])
```

Basic operations on `ScientificDetectors.PreprocessingParameters` instance `obj`:

```julia
size(obj)   # yields the dimensions of the detector
size(obj,k) # yields the `k`-th dimension of the detector
eltype(obj) # yields the floating-point type of the preprocessing data
T(obj)      # convert contents of `obj` to floating-point type `T`
```

"""
struct PreprocessingParameters{T<:AbstractFloat,N,
                               A<:DenseArray{T,N}}
    # Dimensions:
    dims::NTuple{N,Int}

    # Amplitude correction factor (in flux units per ADU):
    a::A

    # Bias correction (in ADU):
    b::A

    # Variance/precision parameters:
    u::A
    v::A

    # Inner constructor provided to force using outer constructors.
    function PreprocessingParameters{T,N,A}(dims::NTuple{N,Int},
                                            a::A,
                                            b::A,
                                            u::A,
                                            v::A) where {T<:AbstractFloat,N,
                                                         A<:DenseArray{T,N}}
        @assert size(a) == dims
        @assert size(b) == dims
        @assert size(u) == dims
        @assert size(v) == dims
        all(x -> isfinite(x) && x ≥ 0, a) ||
            error("some invalid values in amplitude correction")
        all(x -> isfinite(x), b) ||
            error("some invalid values in bias correction")
        all(x -> isfinite(x) && x ≥ 0, u) ||
            error("some invalid values in variance term U")
        all(x -> isfinite(x) && x > 0, v) ||
            error("some invalid values in variance term V")
        return new{T,N,A}(dims, a, b, u, v)
    end
end

# Basic operations on PreprocessingParameters structure.

Base.eltype(::PreprocessingParameters{T}) where {T} = T
Base.size(obj::PreprocessingParameters) = obj.dims
Base.size(obj::PreprocessingParameters, k) = obj.dims[k]
Base.convert(::Type{PreprocessingParameters}, obj::PreprocessingParameters) = obj
Base.convert(::Type{PreprocessingParameters{T}}, obj::PreprocessingParameters) where {T} = T(obj)

for T in (:Float32, :Float64)
    @eval begin
        Base.$T(obj::PreprocessingParameters{$T}) = obj
        Base.$T(obj::PreprocessingParameters{<:Any,N}) where {N} =
            PreprocessingParameters(obj.dims,
                                    convert(Array{$T,N}, obj.a),
                                    convert(Array{$T,N}, obj.b),
                                    convert(Array{$T,N}, obj.u),
                                    convert(Array{$T,N}, obj.v))
    end
end

# Outer constructors for PreprocessingParameters structure.

function PreprocessingParameters(args::AbstractArray{<:Real,N}...) where {N}
    # Stage 0: check arguments.
    length(args) == 4 ||
        error("bad number of preprocessing parameters")
    has_standard_indexing(args...) ||
        error("all arguments must have standard indexing")

    # Stage 1: determine the element type from that of the arguments and
    #          broadcast dimensions.
    T = float(promote_type(map(eltype, args)...))
    dims = bcastdims(map(size, args)...)
    return _stage2(PreprocessingParameters{T,N}, dims, args)
end

# Stage 2: broadcast arguments to the same dimensions and to fast arrays with
#          the given element type.
function _stage2(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 args::NTuple{4,AbstractArray{<:Real,N}}) where {T<:AbstractFloat,N}
    return _stage3(PreprocessingParameters{T,N}, dims,
                   map(x -> fastarray(bcastlazy(T, x, dims)), args)...)
end

# Stage 3: arguments have all same size and element type, compute remaining
#          preprocessing parameters and instanciate structure.
function _stage3(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 a::A, b::A, u::A, v::A)  where {T<:AbstractFloat,N,
                                                 A<:DenseArray{T,N}}
    return PreprocessingParameters{T,N,A}(dims, a, b, u, v)
end
function _stage3(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 a::DenseArray{T,N}, b::DenseArray{T,N}, u::DenseArray{T,N},
                 v::DenseArray{T,N})  where {T<:AbstractFloat,N}
    # Stage 2 failed to produce arrays of same type, convert them to regular Array's.
    A = Array{T,N}
    return PreprocessingParameters{T,N,A}(dims, map(x -> convert(A, x), (a, b, u, v))...)
end

# This version converts calibration parameters to preprocessing parameters.
function PreprocessingParameters(cal::ReducedCalibration{T,N},
                                 bg::Union{Integer,String}=0,
                                 Δt::Real=0) :: PreprocessingParameters{T,N} where {T,N}
    # Check arguments.
    dims = size(cal)
    a, z, g, s, c, cids = cal.a, cal.z, cal.g, cal.s, cal.c, cal.cids
    @assert size(a) == dims
    @assert size(z) == dims
    @assert size(g) == dims
    @assert size(s) == dims
    (isfinite(Δt) && Δt ≥ 0) ||
        error("exposure time must be nonnegative")
    local k::Int
    nc = length(c)
    if nc == 0
        Δt == 0 ||
            error("no time dependent bias specified in calibration data")
        k = 0
    elseif isa(bg, Integer)
        1 ≤ bg ≤ nc ||
            error("out of range index of background source")
        k = bg
    else
        k = -1
        for i in 1:nc
            if cids[i] == bg
                k = i
                break
            end
        end
        k ≥ 1 ||
            error("identifier of background source not found")
    end
    if k > 0
        @assert size(c[k]) == dims
    end

    # Allocate arrays and compute variance terms (first assuming Δt = 0).
    u = similar(a)
    v = similar(a)
    ubad = zero(T) # value of u[i] for defective data
    vbad = one(T)  # idem for v[i] but to avoid division by zero
    @inbounds for i in eachindex(u, v, a, g, s)
        if (isfinite(a[i]) && a[i] > 0 &&
            isfinite(g[i]) && g[i] > 0 &&
            isfinite(s[i]) && s[i] > 0)
            u[i] = g[i]/a[i]
            v[i] = a[i]*g[i]*s[i]^2
        else
            u[i] = ubad
            v[i] = vbad
        end
    end
    if k > 0
        # Account for the contribution of the background and the dark current
        # in the bias and the variance.
        dt = T(Δt)
        b = similar(a)
        @inbounds for i in eachindex(u, v, a, g, s)
            if (u[i] > 0 && isfinite(c[i]) && c[i] ≥ 0)
                cdt = c[i]*dt
                v[i] += a[i]*cdt
                b[i] = z[i] + cdt
            else
                b[i] = z[i]
            end
        end
        return PreprocessingParameters(a, b, u, v)
    else
        return PreprocessingParameters(a, z, u, v)
    end
end

"""

```julia
ScientificDetectors.process(prm, raw, noise=Val(:static)) -> wgt, dat
```

yields a tuple of 2 arrays, `(wgt,dat)`, where `dat` gives the pixel values
while `wgt` gives their respective weights.  both are the result of the
pre-processing of the image `raw` acquired by the detector whose pre-processing
parameters are given by `prm` (an instance of
[`ScientificDetectors.PreprocessingParameters`](@ref)).  The `noise` argument
indicates the model assumed to compute the statistical weights: `Val(:iid)` for
i.i.d. (independent and identically distributed) noise, `Val(:static)` for
static weights independent of `dat` or `Val(:poissonian)` for assuming
Poissonian noise dependent of `dat`.  At least in the 2 first cases, the
statistical weights are given up to a constant factor.

```julia
ScientificDetectors.process(Tao.WeightedArray, prm, raw,
                               noise=Val(:static)) -> wgtimg
```

yields an instance of [`Tao.WeightedArray`](@ref) which is obtained by
pre-processing the image `raw` acquired by the detector whose calibration
parameters are given by `prm`.

The operation can be applied in-place:

```julia
ScientificDetectors.process!(wgt, dat, prm, raw,
                                noise=Val(:static)) -> wgt, dat
```

or

```julia
ScientificDetectors.process(wgtimg, prm, raw,
                               noise=Val(:static)) -> wgtimg
```

to overwrite the contents of `wgt` and `dat` or of `wgtimg` by the result of
the pre-processing.

See also: [`ScientificDetectors.calibrate`](@ref).

"""
function process(prm::PreprocessingParameters{T,N},
                 raw::AbstractArray{<:Number,N},
                 noise::Val = Val(:static)) where {T<:AbstractFloat,N}
    dims = dimensions(raw)
    return process!(Array{T,N}(undef, dims), Array{T,N}(undef, dims),
                    prm, raw, noise)
end

function process!(wgt::AbstractArray, dat::AbstractArray,
                  prm::PreprocessingParameters, raw::AbstractArray)
    return process!(wgt, dat, prm, raw, Val(:static))
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Number,N},
                  ::Val{:iid}) where {T<:AbstractFloat,N}
    dims = dimensions(raw)
    @assert dimensions(wgt) == dims
    @assert dimensions(dat) == dims
    a, b = prm.a, prm.b
    @assert dimensions(prm.a) == dims
    @assert dimensions(prm.b) == dims
    @inbounds @simd for i in eachindex(raw, dat, a, b)
        dat[i] = (T(raw[i]) - b[i])*a[i]
    end
    @inbounds @simd for i in eachindex(wgt)
        wgt[i] = one(T)
    end
    return wgt, dat
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Number,N},
                  ::Val{:static}) where {T<:AbstractFloat,N}
    dims = dimensions(raw)
    @assert dimensions(wgt) == dims
    @assert dimensions(dat) == dims
    a, b, u = prm.a, prm.b, prm.u
    @assert dimensions(prm.a) == dims
    @assert dimensions(prm.b) == dims
    @assert dimensions(prm.u) == dims
    @inbounds @simd for i in eachindex(raw, dat, a, b)
        dat[i] = (T(raw[i]) - b[i])*a[i]
    end
    @inbounds @simd for i in eachindex(wgt, u)
        wgt[i] = u[i]
    end
    return wgt, dat
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Number,N},
                  ::Val{:poissonian}) where {T<:AbstractFloat,N}
    dims = dimensions(raw)
    @assert dimensions(wgt) == dims
    @assert dimensions(dat) == dims
    a, b, u, v = prm.a, prm.b, prm.u, prm.v
    @assert dimensions(prm.a) == dims
    @assert dimensions(prm.b) == dims
    @assert dimensions(prm.u) == dims
    @assert dimensions(prm.v) == dims
    if true
        # Compute dat and wgt separately.
        @inbounds @simd for i in eachindex(raw, dat, a, b)
            dat[i] = (T(raw[i]) - b[i])*a[i]
        end
        @inbounds @simd for i in eachindex(wgt, dat, u, v)
            wgt[i] = u[i]/(v[i] + max(dat[i], zero(T)))
        end
    else
        # Compute dat and wgt jointly.
        @inbounds @simd for i in eachindex(raw, wgt, dat, a, b, u, v)
            d = (T(raw[i]) - b[i])*a[i]
            dat[i] = d
            wgt[i] = u[i]/(v[i] + max(d, zero(T)))
        end
    end
    return wgt, dat
end

@doc @doc(process) process!

end # module
