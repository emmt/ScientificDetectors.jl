module Preprocessing

import ..ScientificDetectors: ReducedCalibration, process, process!

using ..Calibration: detectorbias, detectorgain, detectornoise,
    categories, exposuretimes, currents, find

using ArrayTools

abstract type NoiseModel end
struct RealisticNoise <: NoiseModel end
struct IndependentIdenticallyDistributedNoise <: NoiseModel end
struct StaticNoise <: NoiseModel end

const DEFAULT_NOISE_MODEL = RealisticNoise()

"""

`PreprocessingParameters{T}` stores the pre-processing
parameters with `T` the floating-point type for the computations.

Constructor is called as:

```julia
PreprocessingParameters([T,] a, b, p, q) -> obj
```

where `T` is the floating-point type for the computations (optional, if
unspecified, it is deduced from the element type of the other arguments), `a`
is the amplitude correction factor (in flux units per ADU), `b` is the bias
correction (in ADU), `p` and `q` are variance terms.  Arguments `a`, `b`, `p`
and `q` are pixelwise, they are broadcast to common dimensions (which should be
that of the detector) and their elements converted to the same type `T`.

It is also possible to convert reduced calibration data to preprocessing
parameters:

```julia
PreprocessingParameters(cal::ReducedCalibration,
                        bad=zeros(Bool, size(cal));
                        flat=nothing, flatbg=nothing,
                        bg=nothing, Δt=0) -> obj
```

with `T` the floating point type of the result, `cal` an instance of
[`ReducedCalibration`](@ref), `bad` a boolean mask
indicating the bad pixels, `flat` the identifier of the flat calibration
source, `flatbg` the identifier of the background source for the flat, `bg` the
identifier of the background source and `Δt` the exposure time in seconds.
Identifiers `flat`, `flatbg` and/or `bg` can be `nothing` if irrelevant or an
integer or a string that corresponds to a specific current term in the reduced
calibration data `cal`.

The floating-point type of the result, say `T`, can be explicitly specified:

```julia
PreprocessingParameters{T}(cal, ...) -> obj
```

The pre-processing of pixel `raw[i]` of an image given by the detector leads to
compute the calibrated pixel value `dat[i]` and its corresponding precision
`wgt[i]` as follows:

```julia
dat[i] = (T(raw[i]) - b[i])*a[i]
wgt[i] = p[i]/(max(zero(T), dat[i]) + q[i])
```

where the *realistic* noise model has been assumed.  These operations are
efficiently done by the [`process`](@ref) method.

Basic operations on `PreprocessingParameters` instance `obj`:

```julia
size(obj)   # yields the dimensions of the detector
size(obj,k) # yields the `k`-th dimension of the detector
length(obj) # yields the number of elements of the detector
eltype(obj) # yields the floating-point type of the preprocessing data
T.(obj)     # convert contents of `obj` to floating-point type `T`
```

Also see [`process`](@ref).

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
    p::A
    q::A

    # Inner constructor provided to force using outer constructors.
    function PreprocessingParameters{T,N,A}(dims::NTuple{N,Int},
                                            a::A,
                                            b::A,
                                            p::A,
                                            q::A) where {T<:AbstractFloat,N,
                                                         A<:DenseArray{T,N}}
        @assert size(a) == dims
        @assert size(b) == dims
        @assert size(p) == dims
        @assert size(q) == dims
        all(x -> isfinite(x) && x ≥ 0, a) ||
            error("some invalid values in amplitude correction")
        all(x -> isfinite(x), b) ||
            error("some invalid values in bias correction")
        all(x -> isfinite(x) && x ≥ 0, p) ||
            error("some invalid values in variance term U")
        all(x -> isfinite(x) && x > 0, q) ||
            error("some invalid values in variance term V")
        return new{T,N,A}(dims, a, b, p, q)
    end
end

#
# Outer constructors for PreprocessingParameters structure.
#

# A constructor of an immutable structure can return its argument.
PreprocessingParameters(obj::PreprocessingParameters) = obj
PreprocessingParameters{T}(obj::PreprocessingParameters{T}) where {T} = obj
PreprocessingParameters{T,N}(obj::PreprocessingParameters{T,N}) where {T,N} = obj
PreprocessingParameters{T,N}(obj::PreprocessingParameters{<:Any,N}) where {T,N} =
    PreprocessingParameters{T}(obj)
PreprocessingParameters{T}(obj::PreprocessingParameters{<:Any,N}) where {T<:AbstractFloat,N} =
    PreprocessingParameters(convert(Array{T,N}, obj.a),
                            convert(Array{T,N}, obj.b),
                            convert(Array{T,N}, obj.p),
                            convert(Array{T,N}, obj.q))

function PreprocessingParameters(args::AbstractArray{<:Real,N}...) where {N}
    # Stage 0: Check arguments.
    length(args) == 4 ||
        error("bad number of preprocessing parameters")
    has_standard_indexing(args...) ||
        error("all arguments must have standard indexing")

    # Stage 1: Determine the element type from that of the arguments and
    #          broadcast dimensions.
    T = float(promote_type(map(eltype, args)...))
    dims = bcastdims(map(size, args)...)
    return _stage2(PreprocessingParameters{T,N}, dims, args)
end

# Stage 2: Broadcast arguments to the same dimensions and to fast arrays with
#          the given element type.
function _stage2(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 args::NTuple{4,AbstractArray{<:Real,N}}) where {T<:AbstractFloat,N}
    return _stage3(PreprocessingParameters{T,N}, dims,
                   map(x -> fastarray(bcastlazy(T, x, dims)), args)...)
end

# Stage 3: Arguments have all same size and element type, instanciate
#          structure.  If Stage 2 failed to produce arrays of same type,
#          convert them to regular Array's.
function _stage3(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 a::A, b::A, p::A, q::A) where {T<:AbstractFloat,N,
                                                A<:DenseArray{T,N}}
    return PreprocessingParameters{T,N,A}(dims, a, b, p, q)
end
function _stage3(::Type{<:PreprocessingParameters{T,N}}, dims::NTuple{N,Int},
                 a::AbstractArray{T,N}, b::AbstractArray{T,N}, p::AbstractArray{T,N},
                 q::AbstractArray{T,N})  where {T<:AbstractFloat,N}
    A = Array{T,N}
    return PreprocessingParameters{T,N,A}(dims, map(x -> convert(A, x), (a, b, p, q))...)
end

# This version converts reduced calibration parameters to preprocessing parameters.
PreprocessingParameters(cal::ReducedCalibration{T}, args...; kwds...) where {T} =
    PreprocessingParameters{T}(cal, args...; kwds...)

function PreprocessingParameters{T}(cal::ReducedCalibration{R,N},
                                    bad::AbstractArray{Bool,N} = zeros(Bool, size(cal));
                                    flat::Union{Nothing,Integer,String} = nothing,
                                    flatbg::Union{Nothing,Integer,String} = nothing,
                                    bg::Union{Nothing,Integer,String} = nothing,
                                    Δt::Real=0) where {T<:AbstractFloat,R,N}

    # Get index of flat term and its background.
    jflat = find(cal, flat)
    jflat != 0 ||
        error("`flat` keyword must be specified with a valid identifier/index")
    jflatbg = find(cal, flatbg)
    (flatbg !== nothing && jflatbg == 0 ) &&
        error("invalid identifier/index of background source for the flat")

    # Get index of background term.
    jbg = find(cal, bg)
    (bg !== nothing && jbg == 0 ) &&
        error("invalid identifier/index of background source")

    # Get exposure time.
    (isfinite(Δt) && Δt ≥ 0) ||
        error("exposure time must be nonnegative")
    (jbg != 0 && Δt == 0) &&
        error("no time dependent bias specified in calibration data")

    # Check arguments.
    dims = size(cal)
    z = detectorbias(cal)
    g = detectorgain(cal)
    σ = detectornoise(cal)
    c = currents(cal)
    cat = categories(cal)
    @assert size(bad) == dims
    @assert size(z) == dims
    @assert size(g) == dims
    @assert size(σ) == dims
    @assert size(c[jflat]) == dims
    if jflatbg != 0
        @assert size(c[jflatbg]) == dims
    end
    if jbg != 0
        @assert size(c[jbg]) == dims
    end

    # Compute the flux correction term.  Bad pixels have a[i] = 0.
    a = Array{T}(undef, dims)
    cflat = c[jflat]
    if jflatbg != 0
        cflatbg = c[jflatbg]
        @inbounds for i in eachindex(bad, a, cflat, cflatbg)
            val = cflat[i] - cflatbg[i]
            a[i] = (bad[i] | !isfinite(val) | (val ≤ 0)) ? zero(T) : one(T)/val
        end
    else
        @inbounds for i in eachindex(bad, a, cflat)
            val = cflat[i]
            a[i] = (bad[i] | !isfinite(val) | (val ≤ 0)) ? zero(T) : one(T)/val
        end
    end

    # Compute the bias correction and the variance terms.  Bad pixels have
    # a[i] = 0, b[i] = 0, p[i] = 0 and q[i] = 1, to have zero precision and
    # avoid division by zero.
    b = Array{T}(undef, dims)
    p = Array{T}(undef, dims)
    q = Array{T}(undef, dims)
    if jbg != 0
        cbg = c[jbg]
        dt = T(Δt)
        @inbounds for i in eachindex(bad, a, b, g, p, q, σ, cbg)
            cdt = cbg[i]*dt
            b_i = z[i] + cdt
            p_i = g[i]/a[i]
            q_i = a[i]*(g[i]*σ[i]^2 + cdt)
            if ((a[i] ≤ 0) | !isfinite(b_i) | !(isfinite(p_i) & (p_i ≥ 0)) |
                !(isfinite(q_i) & (q_i > 0)))
                a[i] = zero(T)
                b[i] = zero(T)
                p[i] = zero(T)
                q[i] = one(T)
            else
                b[i] = b_i
                p[i] = p_i
                q[i] = q_i
            end
        end
    else
        @inbounds for i in eachindex(bad, a, b, g, p, q, σ)
            b_i = z[i]
            p_i = g[i]/a[i]
            q_i = a[i]*g[i]*σ[i]^2
            if ((a[i] ≤ 0) | !isfinite(b_i) | !(isfinite(p_i) & (p_i ≥ 0)) |
                !(isfinite(q_i) & (q_i > 0)))
                a[i] = zero(T)
                b[i] = zero(T)
                p[i] = zero(T)
                q[i] = one(T)
            else
                b[i] = b_i
                p[i] = p_i
                q[i] = q_i
            end
        end
    end
    return PreprocessingParameters(a, b, p, q)
end

#
# Basic operations on PreprocessingParameters structure.
#
Base.eltype(::PreprocessingParameters{T}) where {T} = T
Base.size(obj::PreprocessingParameters) = obj.dims
Base.size(obj::PreprocessingParameters, k) = obj.dims[k]
Base.length(obj::PreprocessingParameters) = prod(size(obj))
Base.convert(::Type{PreprocessingParameters}, obj::PreprocessingParameters) = obj
Base.convert(::Type{PreprocessingParameters{T}}, obj::PreprocessingParameters) where {T<:AbstractFloat} = T.(obj)

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::PreprocessingParameters{T}) where {T} = obj
Broadcast.broadcasted(::Type{T}, obj::PreprocessingParameters) where {T<:AbstractFloat} =
    PreprocessingParameters{T}(obj)

"""

```julia
process(prm, raw, noise=RealisticNoise()) -> wgt, dat
```

yields a tuple of 2 arrays, `(wgt,dat)`, where `dat` gives the pixel values
while `wgt` gives their respective weights.  Both are the result of the
pre-processing of the image `raw` acquired by the detector whose pre-processing
parameters are given by `prm` (an instance of
[`PreprocessingParameters`](@ref)).  The `noise` argument indicates the model
assumed to compute the statistical weights:

- `Val(:iid)` for i.i.d. (independent and identically distributed) noise;

- `Val(:static)` for static weights independent of `dat`;

- `Val(:realistic)` for assuming realistic noise dependent of `dat`.

In the 2 first cases, the statistical weights are given up to a constant
factor.

The operation can be applied in-place:

```julia
process!(wgt, dat, prm, raw, noise=Val(:realistic)) -> wgt, dat
```

to overwrite the contents of `wgt` and `dat` by the result of the
pre-processing.  This is useful to avoid re-allocating arrays.

See also: [`calibrate`](@ref).

"""
function process(prm::PreprocessingParameters{T,N},
                 raw::AbstractArray{<:Real,N},
                 noisemodel::NoiseModel =
                 DEFAULT_NOISE_MODEL) where {T<:AbstractFloat,N}
    dims = dimensions(raw)
    return process!(Array{T,N}(undef, dims), Array{T,N}(undef, dims),
                    prm, raw, noisemodel)
end

function process!(wgt::AbstractArray, dat::AbstractArray,
                  prm::PreprocessingParameters, raw::AbstractArray)
    return process!(wgt, dat, prm, raw, DEFAULT_NOISE_MODEL)
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Real,N},
                  ::IndependentIdenticallyDistributedNoise) where {T<:AbstractFloat,N}
    # FIXME: _checkshape(....)
    affinecorrection!(dat, prm.a, prm.b, raw)
    iidweights!(wgt, dat, prm)
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Real,N},
                  ::StaticNoise) where {T<:AbstractFloat,N}
    # FIXME: _checkshape(....)
    affinecorrection!(dat, prm, raw)
    staticweights!(wgt, dat, prm)
end

function process!(wgt::AbstractArray{T,N},
                  dat::AbstractArray{T,N},
                  prm::PreprocessingParameters{T,N},
                  raw::AbstractArray{<:Real,N},
                  ::RealisticNoise) where {T<:AbstractFloat,N}
    affinecorrection!(dat, prm, raw)
    realisticweights!(wgt, dat, prm)
end

@doc @doc(process) process!

#
# It is faster to compute the data and the weights separately.  So we provide 2
# methods for that.
#
# Using Julia `@.` macro  yields a much slower code:
#
#     @. dat = (T(raw) - prm.b)*prm.a
#     @. wgt = prm.p/(prm.q + max(dat, zero(T)))
#

"""

```julia
affinecorrection!(dat, prm, raw) -> dat
```

overwrites `dat` with the affine correction specified by the preprocessing
parameters `prm` and applied to the detector data `raw`.

```julia
affinecorrection!(dat, a, b, raw) -> dat
```

does the same but the affine correction being specified by `a` and `b`.

"""
function affinecorrection!(dat::AbstractArray{T,N},
                           prm::PreprocessingParameters{T,N},
                           raw::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    affinecorrection!(dat, prm.a, prm.b, raw)
end

function affinecorrection!(dat::AbstractArray{T,N},
                           a::AbstractArray{T,N},
                           b::AbstractArray{T,N},
                           raw::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    axes(dat) == axes(a) == axes(b) == axes(raw) || incompatible_indices()
    @inbounds @simd for i in eachindex(dat, raw, a, b)
        dat[i] = (T(raw[i]) - b[i])*a[i]
    end
    return dat
end

"""

```julia
realisticweights!(wgt, dat, prm) -> wgt, dat
```

overwrites `wgt` with realistic weights computed from the pre-processed data
`dat` and using the preprocessing parameters `prm`.

```julia
realisticweights!(wgt, dat, p, q) -> wgt, dat
```

does the same with the preprocessing parameters specified by `p` and `q`.

"""
function realisticweights!(wgt::AbstractArray{T,N},
                           dat::AbstractArray{T,N},
                           prm::PreprocessingParameters{T,N}) where {T<:AbstractFloat,N}
    realisticweights!(wgt, dat, prm.p, prm.q)
end

function realisticweights!(wgt::AbstractArray{T,N},
                           dat::AbstractArray{T,N},
                           p::AbstractArray{T,N},
                           q::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(p) == axes(q) || incompatible_indices()
    @inbounds @simd for i in eachindex(wgt, dat, p, q)
        wgt[i] = p[i]/(q[i] + max(dat[i], zero(T)))
    end
    return wgt, dat
end

"""

```julia
iidweights!(wgt, dat, args...) -> wgt, dat
```

overwrites `wgt` with weights corresponding to independentid
enticallydistributed (i.i.d.) noise, that is fill `wgt` with ones.

"""
function iidweights!(wgt::AbstractArray{T,N},
                     dat::AbstractArray{T,N},
                     prm::PreprocessingParameters{T,N}) where {T<:AbstractFloat,N}
    iidweights!(wgt, dat, prm.p, prm.q)
end

function iidweights!(wgt::AbstractArray{T,N},
                     dat::AbstractArray{T,N},
                     p::AbstractArray{T,N},
                     q::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(p) == axes(q) || incompatible_indices()
    if false
        @inbounds @simd for i in eachindex(wgt)
            wgt[i] = one(T)
        end
    else
        fill!(wgt, one(T))
    end
    return wgt, dat
end

"""

```julia
staticweights!(wgt, dat, prm) -> wgt, dat
```

overwrites `wgt` with static weights that only depend on the preprocessing
parameters `prm` and do not depend on the pre-processed data `dat`.

```julia
staticweights!(wgt, dat, p, q) -> wgt, dat
```

does the same with the preprocessing parameters specified by `p` and `q`.

"""
function staticweights!(wgt::AbstractArray{T,N},
                        dat::AbstractArray{T,N},
                        prm::PreprocessingParameters{T,N}) where {T<:AbstractFloat,N}
    staticweights!(wgt, dat, prm.p, prm.q)
end

function staticweights!(wgt::AbstractArray{T,N},
                        dat::AbstractArray{T,N},
                        p::AbstractArray{T,N},
                        q::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(p) == axes(q) || incompatible_indices()
    if false
        @inbounds @simd for i in eachindex(wgt, p)
            wgt[i] = p[i]
        end
    else
        copyto!(wgt, p)
    end
    return wgt, dat
end

@noinline incompatible_dimensions() =
    throw(DimensionMismatch("incompatible dimensions"))

@noinline incompatible_indices() =
    throw(DimensionMismatch("incompatible indices"))

end # module
