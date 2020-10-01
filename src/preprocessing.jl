module Preprocessing

using ..ScientificDetectors
using ..ScientificDetectors: offset, binning
import ..ScientificDetectors: ReducedCalibration, process, process!,
    regionofinterest, exposuretime

using ..Calibration: detectorbias, detectorgain, detectornoise,
    categories, exposuretimes, currents, find

using ArrayTools

abstract type NoiseModel end
struct RealisticNoise <: NoiseModel end
struct IndependentIdenticallyDistributedNoise <: NoiseModel end
struct StaticNoise <: NoiseModel end

const DEFAULT_NOISE_MODEL = RealisticNoise()

"""

`PreprocessingParameters{T}` stores the pre-processing parameters with `T` the
floating-point type for the computations.

Constructor is called as:

```julia
PreprocessingParameters([roi,] Δt, a, b, q, r)
```

where `roi` is an `N`-tuple of `DetectorAxis` describing the region of interest
(automatically guessed from argument `a` if not specified), `Δt` is the
exposure time (in seconds), `a` is the amplitude correction factor (in flux
units per ADU), `b` is the bias correction (in ADU), `q` and `r` are variance
terms.  Arguments `a`, `b`, `q` and `r` are pixelwise.

It is also possible to convert reduced calibration data to preprocessing
parameters:

```julia
PreprocessingParameters(cal::ReducedCalibration,
                        bad=zeros(Bool, size(cal));
                        flat=nothing, flatbg=nothing,
                        bg=nothing, Δt=0) -> obj
```

with `cal` an instance of [`ReducedCalibration`](@ref), `bad` a boolean mask
indicating the bad pixels, `flat` the identifier of the flat calibration
source, `flatbg` the identifier of the background source for the flat, `bg` the
identifier of the background source and `Δt` the exposure time in seconds.
Identifiers `flat`, `flatbg` and/or `bg` can be `nothing` if irrelevant or an
integer or a string that corresponds to a specific current term in the reduced
calibration data `cal`.

Floating-point type, say `T`, and dimensionality, say `N`, may be specified:

```julia
PreprocessingParameters{T}(args...; kwds...)
PreprocessingParameters{T,N}(args...; kwds...)
```

The pre-processing of pixel `raw[i]` of an image given by the detector leads to
compute the calibrated pixel value `dat[i]` and its corresponding precision
`wgt[i]` as follows:

```julia
dat[i] = (T(raw[i]) - b[i])*a[i]
wgt[i] = q[i]/(max(zero(T), dat[i]) + r[i])
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
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::NTuple{N,DetectorAxis}

    # Exposure time (in seconds).
    Δt::Float64

    # Amplitude correction factor (in flux units per ADU):
    a::A

    # Bias correction (in ADU):
    b::A

    # Variance/precision parameters:
    q::A
    r::A

    # Inner constructor provided to force using outer constructors.
    function PreprocessingParameters{T,N,A}(roi::NTuple{N,DetectorAxis},
                                            Δt::Real,
                                            a::A,
                                            b::A,
                                            q::A,
                                            r::A) where {T<:AbstractFloat,N,
                                                         A<:DenseArray{T,N}}
        @assert isfinite(Δt) && Δt ≥ 0
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(a) == dims
        @assert size(b) == dims
        @assert size(q) == dims
        @assert size(r) == dims
        all(x -> isfinite(x) && x ≥ 0, a) ||
            error("some invalid values in amplitude correction")
        all(x -> isfinite(x), b) ||
            error("some invalid values in bias correction")
        all(x -> isfinite(x) && x ≥ 0, q) ||
            error("some invalid values in variance term U")
        all(x -> isfinite(x) && x > 0, r) ||
            error("some invalid values in variance term V")
        return new{T,N,A}(roi, Δt, a, b, q, r)
    end
end

#
# Outer constructors for construction given all fields.
#
function PreprocessingParameters(roi::NTuple{N,DetectorAxis},
                                 Δt::Real,
                                 a::AbstractArray{<:Real,N},
                                 b::AbstractArray{<:Real,N},
                                 q::AbstractArray{<:Real,N},
                                 r::AbstractArray{<:Real,N}) where {N}
    T = float(promote_type(eltype(a), eltype(b), eltype(q), eltype(r)))
    PreprocessingParameters(roi, Δt, a, b, q, r)
end

function PreprocessingParameters{T}(roi::NTuple{N,DetectorAxis},
                                    Δt::Real,
                                    a::AbstractArray{<:Real,N},
                                    b::AbstractArray{<:Real,N},
                                    q::AbstractArray{<:Real,N},
                                    r::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    PreprocessingParameters{T,N}(roi, Δt,
                                 convert(Array{T,N}, a),
                                 convert(Array{T,N}, b),
                                 convert(Array{T,N}, q),
                                 convert(Array{T,N}, r))
end

function PreprocessingParameters{T,N}(roi::NTuple{N,DetectorAxis},
                                      Δt::Real,
                                      a::AbstractArray{<:Real,N},
                                      b::AbstractArray{<:Real,N},
                                      q::AbstractArray{<:Real,N},
                                      r::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    PreprocessingParameters{T}(roi, Δt, a, b, q, r)
end

#
# Simple outer constructors for conversion.
#
PreprocessingParameters(obj::PreprocessingParameters) = obj
PreprocessingParameters{T}(obj::PreprocessingParameters{T}) where {T} = obj
PreprocessingParameters{T}(obj::PreprocessingParameters{<:Any,N}) where {T<:AbstractFloat,N} =
    PreprocessingParameters(obj.roi, obj.Δt, obj.a, obj.b, obj.q, obj.r)
PreprocessingParameters{T,N}(obj::PreprocessingParameters{T,N}) where {T,N} = obj
PreprocessingParameters{T,N}(obj::PreprocessingParameters{<:Any,N}) where {T,N} =
    PreprocessingParameters{T}(obj)

#
# Getters.
#
regionofinterest(obj::PreprocessingParameters) = obj.roi
exposuretime(obj::PreprocessingParameters) = obj.Δt

#
# Basic operations on PreprocessingParameters structure.
#
Base.eltype(::PreprocessingParameters{T}) where {T} = T
Base.size(obj::PreprocessingParameters) = size(regionofinterest(obj))
Base.size(obj::PreprocessingParameters, i) = size(regionofinterest(obj), i)
Base.length(obj::PreprocessingParameters) = prod(size(obj))
Base.convert(::Type{T}, obj::PreprocessingParameters) where {T<:PreprocessingParameters} =
    T(obj)

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::PreprocessingParameters) where {T<:AbstractFloat} =
    PreprocessingParameters{T}(obj)

Base.show(io::IO, obj::PreprocessingParameters{T,N}) where {T,N} = begin
    join(io, size(obj), "×")
    print(io, "PreprocessingParameters{$T,$N}: Δt = ", exposuretime(obj), " s")
end


#
# More complex outer constructors for PreprocessingParameters structure.
#

# Provide a ROI if not specified.
function PreprocessingParameters(Δt::Real,
                                 a::AbstractArray, b::AbstractArray,
                                 q::AbstractArray, r::AbstractArray)
    PreprocessingParameters(map(DetectorAxis, size(f)), Δt, a, b, q, r)
end

function PreprocessingParameters{T}(Δt::Real,
                                    a::AbstractArray, b::AbstractArray,
                                    q::AbstractArray, r::AbstractArray) where {T}
    PreprocessingParameters{T}(map(DetectorAxis, size(a)), Δt, a, b, q, r)
end

function PreprocessingParameters{T,N}(Δt::Real,
                                      a::AbstractArray, b::AbstractArray,
                                      q::AbstractArray, r::AbstractArray) where {T,N}
    PreprocessingParameters{T,N}(map(DetectorAxis, size(a)), Δt, a, b, q, r)
end

# Provide parameters T and N.
function PreprocessingParameters(roi::NTuple{N,DetectorAxis}, Δt::Real,
                                 a::AbstractArray, b::AbstractArray,
                                 q::AbstractArray, r::AbstractArray) where {N}
    T = float(promote_eltype(a, b, q, r))
    PreprocessingParameters{T,N}(roi, Δt, a, b, q, r)
end

# Provide parameter N.
function PreprocessingParameters{T}(roi::NTuple{N,DetectorAxis}, Δt::Real,
                                    a::AbstractArray, b::AbstractArray,
                                    q::AbstractArray, r::AbstractArray) where {T,N}
    PreprocessingParameters{T,N}(roi, Δt, a, b, q, r)
end

function PreprocessingParameters{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                      a::AbstractArray, b::AbstractArray,
                                      q::AbstractArray, r::AbstractArray) where {T,N}
    T <: AbstractFloat || error("parameter `T` must be a floating-point type")
    length(roi) == N || error("ROI has incompatible number of dimensions")
    dims = size(roi)
    function fixarray(A::AbstractArray)
        Base.has_offset_axes(A) && error("array has non-standard indexing")
        eltype(A) <: Real || error("array has incompatible element type")
        ndims(A) == N || error("array has incompatible number of dimensions")
        size(A) == dims ||
            throw(DimensionMismatch("array has incompatible dimensions"))
        return convert(Array{T,N}, A)
    end
    PreprocessingParameters{T,N}(roi, Δt,
                                 fixarray(a), fixarray(b),
                                 fixarray(q), fixarray(r))
end

# These versions manage to directly call the inner constructor.
function PreprocessingParameters(roi::NTuple{N,DetectorAxis},
                                 Δt::Real, a::A, b::A, q::A,
                                 r::A) where {T<:AbstractFloat,N,
                                              A<:DenseArray{T,N}}
    PreprocessingParameters{T,N,A}(roi, Δt, a, b, q, r)
end

function PreprocessingParameters{T}(roi::NTuple{N,DetectorAxis},
                                    Δt::Real, a::A, b::A, q::A,
                                    r::A) where {T<:AbstractFloat,N,
                                                 A<:DenseArray{T,N}}
    PreprocessingParameters{T,N,A}(roi, Δt, a, b, q, r)
end

function PreprocessingParameters{T,N}(roi::NTuple{N,DetectorAxis},
                                      Δt::Real, a::A, b::A, q::A,
                                      r::A) where {T<:AbstractFloat,N,
                                                   A<:DenseArray{T,N}}
    PreprocessingParameters{T,N,A}(roi, Δt, a, b, q, r)
end

# This version converts reduced calibration parameters to preprocessing parameters.
PreprocessingParameters(cal::ReducedCalibration{T}, args...; kwds...) where {T} =
    PreprocessingParameters{T}(cal, args...; kwds...)

PreprocessingParameters{T,N}(cal::ReducedCalibration{R,N}, args...; kwds...) where {T,R,N} =
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
    # a[i] = 0, b[i] = 0, q[i] = 0 and r[i] = 1, to have zero precision and
    # avoid division by zero.
    b = Array{T}(undef, dims)
    q = Array{T}(undef, dims)
    r = Array{T}(undef, dims)
    if jbg != 0
        cbg = c[jbg]
        dt = T(Δt)
        @inbounds for i in eachindex(bad, a, b, g, q, r, σ, cbg)
            cdt = cbg[i]*dt
            b_i = z[i] + cdt
            q_i = g[i]/a[i]
            r_i = a[i]*(g[i]*σ[i]^2 + cdt)
            if ((a[i] ≤ 0) | !isfinite(b_i) | !(isfinite(q_i) & (q_i ≥ 0)) |
                !(isfinite(r_i) & (r_i > 0)))
                a[i] = zero(T)
                b[i] = zero(T)
                q[i] = zero(T)
                r[i] = one(T)
            else
                b[i] = b_i
                q[i] = q_i
                r[i] = r_i
            end
        end
    else
        @inbounds for i in eachindex(bad, a, b, g, q, r, σ)
            b_i = z[i]
            q_i = g[i]/a[i]
            r_i = a[i]*g[i]*σ[i]^2
            if ((a[i] ≤ 0) | !isfinite(b_i) | !(isfinite(q_i) & (q_i ≥ 0)) |
                !(isfinite(r_i) & (r_i > 0)))
                a[i] = zero(T)
                b[i] = zero(T)
                q[i] = zero(T)
                r[i] = one(T)
            else
                b[i] = b_i
                q[i] = q_i
                r[i] = r_i
            end
        end
    end
    return PreprocessingParameters(regionofinterest(cal), Δt, a, b, q, r)
end

PreprocessingParameters(cal::SimpleCalibration{T}, args...; kwds...) where {T} =
    PreprocessingParameters{T}(cal, args...; kwds...)

function PreprocessingParameters{T}(cal::SimpleCalibration{R,N},
                                    bad::AbstractArray{Bool,N} = zeros(Bool, size(cal))
                                    ) where {T<:AbstractFloat,R,N}

    # Get exposure time.
    Δt = exposuretime(cal)
    (isfinite(Δt) && Δt ≥ 0) || error("exposure time must be nonnegative")

    # Check arguments.
    dims = size(cal)
    a = copy(cal.a)
    b = copy(cal.b)
    g = cal.g
    σ = cal.σ
    @assert size(bad) == dims
    @assert size(a) == dims
    @assert size(b) == dims
    @assert size(g) == dims
    @assert size(σ) == dims

    # Compute the variance terms.  Bad pixels have a[i] = 0, b[i] = 0, q[i] = 0
    # and r[i] = 1, to have zero precision and avoid division by zero.
    q = Array{T}(undef, dims)
    r = Array{T}(undef, dims)
    @inbounds for j in eachindex(bad, a, b, g, σ, q, r)
        if (bad[j] || !isfinite(a[j]) || a[j] ≤ 0 || !isfinite(b[j]) ||
            !isfinite(g[j]) || g[j] ≤ 0 || !isfinite(σ[j]) || σ[j] ≤ 0)
            a[j] = zero(T)
            b[j] = zero(T)
            q[j] = zero(T)
            r[j] = one(T)
        else
            q[j] = g[j]/a[j]
            r[j] = a[j]*g[j]*σ[j]^2
        end
    end
    return PreprocessingParameters(regionofinterest(cal), Δt, a, b, q, r)
end

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
    dims = standard_size(raw)
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
#     @. wgt = prm.q/(prm.r + max(dat, zero(T)))
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
realisticweights!(wgt, dat, q, r) -> wgt, dat
```

does the same with the preprocessing parameters specified by `q` and `r`.

"""
function realisticweights!(wgt::AbstractArray{T,N},
                           dat::AbstractArray{T,N},
                           prm::PreprocessingParameters{T,N}) where {T<:AbstractFloat,N}
    realisticweights!(wgt, dat, prm.q, prm.r)
end

function realisticweights!(wgt::AbstractArray{T,N},
                           dat::AbstractArray{T,N},
                           q::AbstractArray{T,N},
                           r::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(q) == axes(r) || incompatible_indices()
    @inbounds @simd for i in eachindex(wgt, dat, q, r)
        wgt[i] = q[i]/(r[i] + max(dat[i], zero(T)))
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
    iidweights!(wgt, dat, prm.q, prm.r)
end

function iidweights!(wgt::AbstractArray{T,N},
                     dat::AbstractArray{T,N},
                     q::AbstractArray{T,N},
                     r::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(q) == axes(r) || incompatible_indices()
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
staticweights!(wgt, dat, q, r) -> wgt, dat
```

does the same with the preprocessing parameters specified by `q` and `r`.

"""
function staticweights!(wgt::AbstractArray{T,N},
                        dat::AbstractArray{T,N},
                        prm::PreprocessingParameters{T,N}) where {T<:AbstractFloat,N}
    staticweights!(wgt, dat, prm.q, prm.r)
end

function staticweights!(wgt::AbstractArray{T,N},
                        dat::AbstractArray{T,N},
                        q::AbstractArray{T,N},
                        r::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    axes(wgt) == axes(dat) == axes(q) == axes(r) || incompatible_indices()
    if false
        @inbounds @simd for i in eachindex(wgt, q)
            wgt[i] = q[i]
        end
    else
        copyto!(wgt, q)
    end
    return wgt, dat
end

@noinline incompatible_dimensions() =
    throw(DimensionMismatch("incompatible dimensions"))

@noinline incompatible_indices() =
    throw(DimensionMismatch("incompatible indices"))

end # module
