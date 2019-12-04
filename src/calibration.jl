module Calibration

using ..ScientificDetectors
using ..ScientificDetectors: offset, binning
import ..ScientificDetectors: regionofinterest

using Statistics
using OptimPackNextGen

const Colons{N} = NTuple{N,Colon}

# Union of acceptable identifer types.
const Identifiers = Union{AbstractString,Symbol,Integer}

"""
```julia
identifier(key) -> str
```

converts `key` into a string identifier.  Argument `key` can be of any type part
of the union `Identifiers` (a string, a symbol or an integer).

"""
identifier(key::String) = key
identifier(key::AbstractString) = String(key)
identifier(key::Integer) = string("#",key)
identifier(key::Symbol) = String(key)

@doc @doc(identifier) Identifiers

"""

`ReducedCalibration{T}` stores the calibration parameters with `T` the
floating-point type for the computations.

Constructor is called as:

```julia
ReducedCalibration([roi,] f, z, g, σ, args...; kwds...) -> cal
```

where `roi` is an `N`-tuple of `DetectorAxis` describing the region of interest
(automatically guessed from argument `f` if not specified), `f` is the
co-log-likelihood, `z` is the *zero level* that is the constant bias set by the
analog to digital converter (in ADU), `g` is the detector gain (in electrons
per ADU) and `σ` is the standard deviation of the readout noise (in ADU/frame).
Arguments `f`, `z`, `g` and `σ` are pixelwise.

Additional arguments `args...` can be:

- Key-value pairs like `"cat1" => c1`, `:cat2 => c2`, ... of category
  identifiers and arrays corresponding to current terms like the dark current
  or any background flux (in ADU/second).  Arguments `c1`, `c2`, ... are
  assumed to be pixelwise.

- Two arguments: `c = [c1, c2, ...]` and `cat = ["cat1", "cat2", ...]`
  respectively a vector of current terms and of corresponding category
  identifiers.

Floating-point type, say `T`, and dimensionality, say `N`, may be specified:

```julia
ReducedCalibration{T}([roi,] f, z, g, σ, args...; kwds...) -> cal
ReducedCalibration{T,N}([roi,] f, z, g, σ, args...; kwds...) -> cal
```

Basic operations on `ReducedCalibration` instance `obj`:

```julia
size(obj)       # yields the dimensions of the detector
size(obj,k)     # yields the `k`-th dimension of the detector
length(obj)     # yields the number of elements of the detector
eltype(obj)     # yields the floating-point type of the calibration data
T.(obj)         # convert contents of `obj` to floating-point type `T`
```

Other implemented methods (must be imported or prefixed by
`Calibration`):

```julia
cologlikelihood(obj) # yields th co-log-likelihood map
detectorbias(obj)    # yields the constant detector bias (in ADU)
detectorgain(obj)    # yields the detector gain (in e-/ADU)
detectornoise(obj)   # yields the standard deviation of the detector noise (in ADU)
currents(obj)        # yields all the current terms
current(obj, k)      # yields the k-th current term (in ADU/s)
categories(obj)      # yields all the names of the current terms
category(obj, k)     # yields the name of the k-th current term
```

"""
struct ReducedCalibration{T<:AbstractFloat,N}
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::NTuple{N,DetectorAxis}

    # Co-log-likelihood.
    f::Array{T,N}

    # Zero-level (constant bias in ADU):
    z::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise (in ADU/frame):
    σ::Array{T,N}

    # Time dependent bias, e.g. dark current and background flux, (in
    # ADU/second), may be empty or zero-filled:
    c::Vector{Array{T,N}}

    # Categories of the different sources responsible of the different
    # time-dependent bias terms.
    cat::Vector{String}

    # Inner constructor provided to force using outer constructors.
    function ReducedCalibration{T,N}(roi::NTuple{N,DetectorAxis},
                                     f::Array{T,N},
                                     z::Array{T,N},
                                     g::Array{T,N},
                                     σ::Array{T,N},
                                     c::Vector{Array{T,N}},
                                     cat::Vector{String};
                                     check::Bool = false
                                     ) where {T<:AbstractFloat,N}
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(f) == dims
        @assert size(z) == dims
        @assert size(g) == dims
        @assert size(σ) == dims
        @assert length(cat) == length(c)
        for k ∈ eachindex(c)
            @assert size(c[k]) == dims
        end
        obj = new{T,N}(roi, f, z, g, σ, c, cat)
        check && checkvalues(obj)
        return obj
    end
end

regionofinterest(obj::ReducedCalibration) = obj.roi
cologlikelihood(obj::ReducedCalibration) = obj.f
detectorbias(obj::ReducedCalibration) = obj.z
detectorgain(obj::ReducedCalibration) = obj.g
detectornoise(obj::ReducedCalibration) = obj.σ
currents(obj::ReducedCalibration) = obj.c
current(obj::ReducedCalibration, k::Integer) = getindex(currents(obj), k)
categories(obj::ReducedCalibration) = obj.cat
category(obj::ReducedCalibration, k::Integer) = getindex(categories(obj), k)

#
# Outer constructors for ReducedCalibration structure.
#

# A constructor of an immutable structure can return its argument.
ReducedCalibration(obj::ReducedCalibration) = obj
ReducedCalibration{T}(obj::ReducedCalibration{T}) where {T} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{T,N}) where {T,N} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T}(obj)
ReducedCalibration{T}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T,N}(obj.roi,
                            convert(Array{T,N}, obj.f),
                            convert(Array{T,N}, obj.z),
                            convert(Array{T,N}, obj.g),
                            convert(Array{T,N}, obj.σ),
                            map(x -> convert(Array{T,N}, x), obj.c),
                            obj.cat)

# Provide a ROI if not specified and parse current terms.
function ReducedCalibration(f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(map(DetectorAxis, size(f)), f, z, g, σ,
                       _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T}(f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(map(DetectorAxis, size(f)), f, z, g, σ,
                          _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T,N}(f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(map(DetectorAxis, size(f)), f, z, g, σ,
                            _getcurrents(args...)...; kwds...)
end

# Parse current terms.
function ReducedCalibration(roi::Tuple{Vararg{DetectorAxis}},
                            f::AbstractArray, z::AbstractArray,
                            g::AbstractArray, σ::AbstractArray,
                            args...; kwds...)
    ReducedCalibration(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T}(roi::Tuple{Vararg{DetectorAxis}},
                               f::AbstractArray, z::AbstractArray,
                               g::AbstractArray, σ::AbstractArray,
                               args...; kwds...) where {T}
    ReducedCalibration{T}(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray, z::AbstractArray,
                                 g::AbstractArray, σ::AbstractArray,
                                 args...; kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, _getcurrents(args...)...; kwds...)
end

function ReducedCalibration(roi::NTuple{N,DetectorAxis},
                            f::AbstractArray,
                            z::AbstractArray,
                            g::AbstractArray,
                            σ::AbstractArray,
                            c::AbstractVector{<:AbstractArray},
                            cat::AbstractVector{<:Identifiers};
                            kwds...) where {N}
    T = float(promote_type(eltype(f), eltype(z), eltype(g), eltype(σ),
                           _promote_eltype(c)))
    ReducedCalibration{T,N}(roi, f, z, g, σ, c, cat; kwds...)
end

function ReducedCalibration{T}(roi::NTuple{N,DetectorAxis},
                               f::AbstractArray,
                               z::AbstractArray,
                               g::AbstractArray,
                               σ::AbstractArray,
                               c::AbstractVector{<:AbstractArray},
                               cat::AbstractVector{<:Identifiers};
                               kwds...) where {T,N}
    ReducedCalibration{T,N}(roi, f, z, g, σ, c, cat; kwds...)
end

function ReducedCalibration{T,N}(roi::Tuple{Vararg{DetectorAxis}},
                                 f::AbstractArray,
                                 z::AbstractArray,
                                 g::AbstractArray,
                                 σ::AbstractArray,
                                 c::AbstractVector{<:AbstractArray},
                                 cat::AbstractVector{<:Identifiers};
                                 kwds...) where {T,N}
    T <: AbstractFloat || error("parameter `T` must be a floating-point type")
    length(roi) == N || error("ROI has incompatible number of dimensions")
    length(cat) == length(c) || error("incompatible number of categories")
    dims = size(roi)

    function fixarray(A::AbstractArray)
        Base.has_offset_axes(A) && error("array has non-standard indexing")
        eltype(A) <: Real || error("array has incompatible element type")
        ndims(A) == N || error("array has incompatible number of dimensions")
        size(A) == dims ||
            throw(DimensionMismatch("array has incompatible dimensions"))
        return convert(Array{T,N}, A)
    end
    ReducedCalibration{T,N}(roi,
                            fixarray(f),
                            fixarray(z),
                            fixarray(g),
                            fixarray(σ),
                            map(fixarray, c),
                            map(identifier, cat); kwds...)
end

# Convert pairs like "key1"=>arr1, :key2=>arr2, ... in a list of
# arrays and a list of identifiers.
_getcurrents(args::Pair{<:Union{AbstractString,Symbol},<:AbstractArray}...) =
    (collect(map(x -> x[2], args)),
     collect(map(x -> identifier(x[1]), args)))
_getcurrents() = Int8[], String[]
_getcurrents(c::AbstractVector{<:AbstractArray}, cat::AbstractVector) =
    (c, cat)

"""
```julia
find(obj, key) -> j
```

yields the index `j` of the current term in reduced calibration data which match `key`
or `0` if not found.

"""
find(obj::ReducedCalibration, key::Nothing) = 0

function find(obj::ReducedCalibration, key::AbstractString)
    cat = categories(obj)
    n = 0
    j = 0
    for i in 1:length(cat)
        if cat[i] == key
            j = i
            n += 1
        end
    end
    n > 1 && error("non-unique category identifier")
    return j
end

find(obj::ReducedCalibration, j::Integer) =
    (1 ≤ j ≤ length(categories(obj)) ? Int(j) : 0)

"""
```julia
checkvalues(obj)
```

throws an error if some values in the reduced calibration object `obj` are
invalid.

"""
function checkvalues(cal::ReducedCalibration)
    f, z, g, σ, c = cal.f, cal.z, cal.g, cal.σ, cal.c
    dims = size(cal)
    @assert size(f) == dims
    @assert size(z) == dims
    @assert size(g) == dims
    @assert size(σ) == dims
    for k ∈ eachindex(c)
        @assert size(c[k]) == dims
        all(x -> isfinite(x) && x ≥ 0, c[k]) ||
            error("some invalid values in time-dependent bias")
    end
    all(x -> isfinite(x), z) ||
        error("some invalid values in constant bias")
    all(x -> isfinite(x) && x ≥ 0, g) ||
        error("some invalid values in detector gain")
    all(x -> isfinite(x) && x ≥ 0, σ) ||
        error("some invalid values in readout noise")
end

# Same as ArrayTools.promote_eltype but for a vector of arrays.  Using a
# recursion is the fastest method.
function _promote_eltype(x::AbstractVector{<:AbstractArray})
    n = length(x)
    @assert n ≥ 1
    return _promote_eltype((@inbounds eltype(x[n])), x, n - 1)
end
_promote_eltype(T::Type, x::AbstractVector{<:AbstractArray}, n::Int) =
    (n < 1 ? T :
     _promote_eltype(promote_type(T, (@inbounds eltype(x[n]))), x, n - 1))

#
# Basic operations on ReducedCalibration structure.
#
Base.eltype(::ReducedCalibration{T}) where {T} = T
Base.size(obj::ReducedCalibration) = size(regionofinterest(obj))
Base.size(obj::ReducedCalibration, i) = size(regionofinterest(obj), i)
Base.length(obj::ReducedCalibration) = prod(size(obj))
Base.convert(::Type{T}, obj::ReducedCalibration) where {T<:ReducedCalibration} =
    T(obj)

Base.show(io::IO, obj::ReducedCalibration{T,N}) where {T,N} = begin
    print(io, "ReducedCalibration{$T,$N}: ", join(size(obj),"×"))
    for i in 1:length(categories(obj))
        print(io, "\n - cat", i, ": \"", identifier(category(obj,i)), "\"")
    end
end

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::ReducedCalibration) where {T<:AbstractFloat} =
    ReducedCalibration{T}(obj)

#------------------------------------------------------------------------------

"""
```julia
uniquecategories(A) -> cat, uid
```

given a vector `A` of identifiers or keys, yields the corresponding category
indices `cat` and unique identifers `uid` such that `uid[cat[i]]` is the unique
identifier corresponding to `A[i]`.  The elements of `A` can be of any type
part of the union `Identifiers` (strings, symbols or integers).

"""
function uniquecategories(A::AbstractVector{K}) where {K<:Identifiers}
    # Use a dictionary to collect a unique list of keys and then to store the
    # corresponding unique type number.
    dict = Dict{K,Int}()
    for key in A
        dict[key] = 1
    end
    l = 0
    for key in sort(collect(keys(dict)))
        l += 1
        dict[key] = l
    end

    cat = Vector{Int}(undef, length(A))
    uid = Vector{String}(undef, length(dict))
    i = 0
    for key in A
        i += 1
        cat[i] = dict[key]
    end
    for key in keys(dict)
        uid[dict[key]] = identifier(key)
    end
    return cat, uid
end

# Structure used to store the parameters of a single pixel.
mutable struct FitResult{T}
    f::T         # figure of merit
    z::T         # bias
    g::T         # gain
    u::T         # variance of readout-noise divided by gain
    c::Vector{T} # contributions of the different sources
end

# Structure used to store all calibration data.
"""

```julia
CalibrationData(D, keys, Δt) -> obj
```

yields an object which stores detector calibration data.  Argument `D` is a
vector of detector data frames, `keys` and `Δt` respectively specify the
identifier and exposure time of the corresponding data frame.  The keys can be
integers, symbols or strings.  A given key uniquely identify the category of
the corresponding data frame. Exposure times are in seconds.

```julia
numberofdataframes(obj)  # yields the number of data frames
numberofcategories(obj)  # yields the number of different categories
dataframes(obj)          # yields the vector of data frames
dataframe(obj, i)        # yields the i-th data frame
categories(obj)          # yields the category indices of the data frames
category(obj, i)         # yields the category index of the i-th data frame
exposuretimes(obj)       # yields the exposure times of the data frames
exposuretime(obj, i)     # yields the exposure time of the i-th data frame
uniqueidentifiers(obj)   # yields the list of unique identifiers of categories
uniqueidentifier(obj, l) # yields the l-th unique identifier of categories
```

"""
struct CalibrationData{P<:Real,N,T<:AbstractFloat}
    dims::NTuple{N,Int}      # dimensions of frames
    data::Vector{Array{P,N}} # data[i][j] is j-th pixel of i-th frame
    Δt::Vector{T}            # Δt[i] yields the exposure time of i-th frame
    cat::Vector{Int}         # cat[i] yields the category index of i-th frame
    uid::Vector{String}      # uid[l] is the unique identifer of l-th
                             # calibration category
    function CalibrationData{P,N,T}(data::AbstractVector{Array{P,N}},
                                    keys::AbstractVector{<:Identifiers},
                                    Δt::AbstractVector{T}
                                    ) where {P<:Real,N,T<:AbstractFloat}
        nframes = length(data)
        @assert nframes > 0
        @assert length(keys) == nframes
        @assert length(Δt) == nframes
        dims = size(first(data))
        for A in data
            @assert size(A) == dims
        end
        @assert minimum(Δt) ≥ 0

        cat, uid = uniquecategories(keys)
        ntypes = maximum(cat)

        return new{P,N,T}(dims,
                          convert(Vector{Array{P,N}}, data),
                          convert(Vector{T}, Δt), cat, uid)
    end
end

Base.size(obj::CalibrationData) = obj.dims
Base.size(obj::CalibrationData{P,N,T}, d::Integer) where {P,N,T} =
    (d < 1 ? error("invalid dimension index") :
     d ≤ N ? size(obj)[d] : 1)

dataframes(obj::CalibrationData) = obj.data
categories(obj::CalibrationData) = obj.cat
exposuretimes(obj::CalibrationData) = obj.Δt
uniqueidentifiers(obj::CalibrationData) = obj.uid

numberofdataframes(obj::CalibrationData) = length(dataframes(obj))
numberofcategories(obj::CalibrationData) = length(uniqueidentifiers(obj))

dataframe(obj::CalibrationData, i::Integer) = getindex(dataframes(obj), i)
category(obj::CalibrationData, i::Integer) = getindex(categories(obj), i)
exposuretime(obj::CalibrationData, i::Integer) = getindex(exposuretime(obj), i)
uniqueidentifier(obj::CalibrationData, l::Integer) =
    getindex(uniqueidentifiers(obj), l)

"""
```julia
ReducedCalibration(cal) -> redcal
```

fit the detector parameters in calibration data `cal`.

"""
function ReducedCalibration(cal::CalibrationData{P,N,T}) where {P,N,T}
    nframes = numberofdataframes(cal)
    ntypes = numberofcategories(cal)
    dat = dataframes(cal)
    cat = categories(cal)
    Δt = exposuretimes(cal)
    uid = uniqueidentifiers(cal)
    dims = size(cal)
    @assert length(uid) == ntypes
    @assert length(Δt) == nframes
    @assert length(dat) == nframes
    @assert length(cat) == nframes

    # Check exposure times.
    flag = false
    for i in 1:nframes
        if !isfinite(Δt[i]) || Δt[i] < 0
            error("invalid exposure time(s)")
        end
        if Δt[i] > 0
            flag = true
        end
    end
    if !flag
        error("no non-zero exposure times!")
    end

    # Allocate output and workspaces.
    out = ReducedCalibration(
        Array{T,N}(undef, dims), # f
        Array{T,N}(undef, dims), # z
        Array{T,N}(undef, dims), # g
        Array{T,N}(undef, dims), # σ
        [Array{T,N}(undef, dims) for k in 1:ntypes], # c
        uid)
    d = Array{T,1}(undef, nframes)
    res = FitResult{T}(Inf,NaN,NaN,NaN,
                       fill!(Array{T}(undef, ntypes), NaN));

    # Fit every pixel.
    len = prod(dims)
    for j in 1:len
        # Collect the pixel data.
        for i in 1:nframes
            d[i] = cal.data[i][j]
        end

        # Fit the detector parameters and save them.
        fit!(res, d, cat, Δt)
        for l in 1:ntypes
            out.c[l][j] = res.c[l]
        end
        out.f[j] = res.f
        out.z[j] = res.z
        out.g[j] = res.g
        out.σ[j] = sqrt(res.u/res.g)
    end

    # Return reduced calibration data.
    return out
end

function fit!(res::FitResult{T}, d::Vector{T},
              cat::Vector{Int}, Δt::Vector{T};
              umin::Real = 1e-20) where {T<:AbstractFloat}
    ntypes = length(res.c)
    nframes = length(d)
    @assert length(cat) == nframes
    @assert length(Δt) == nframes

    # Initial weights are 1/(Δt + τ) with τ a small value.
    τ = leastpositive(Δt)/10
    τ > 0 || error("no non-zero exposure times")
    w = Array{T}(undef, nframes)
    update_w!(w, Δt, τ)

    # Initial bias.
    z = minimum(d)

    # Initial current terms are given by a simple constrained linear
    # least-squares fit.
    a = fill!(Array{T,1}(undef, ntypes), 0)
    b = fill!(Array{T,1}(undef, ntypes), 0)
    @inbounds for i in 1:nframes
        l = cat[i]
        a[l] += w[i]*Δt[i]^2
        b[l] += w[i]*Δt[i]*(d[i] - z)
    end
    c = Array{T,1}(undef, ntypes)
    @inbounds for l in 1:ntypes
        c[l] = (b[l] > 0 ? b[l]/a[l] : zero(T))
    end

    # Initial value of u ≡ g⋅σ² is a strictly positive value which is small
    # compared to c⋅Δt.
    cΔt = Array{T}(undef, nframes)
    update_cΔt!(cΔt, c, Δt, cat)
    u = leastpositive(cΔt)/10

    # Initialize initial variables and bounds.
    x = Array{T}(undef, ntypes+1)
    xmin = Array{T}(undef, ntypes+1)
    @inbounds for l in 1:ntypes
        x[l] = c[l]
        xmin[l] = zero(T)
    end
    x[end] = u
    xmin[end] = umin

    # Initialize result so as to store best solution so far.
    res.f = Inf
    res.z = NaN
    res.g = NaN
    res.u = NaN
    fill!(res.c, NaN)

    # Allocate workspaces r for the residuals.
    r = Array{T}(undef, nframes)

    # Define the objective function as a closure to share workspaces and data.
    function fg!(x::Vector{T}, gx::Vector{T})
        # Extract parameters.
        @assert length(x) == length(gx) == ntypes + 1
        @inbounds for l in 1:ntypes
            c[l] = x[l]
        end
        u = x[end]

        # Compute the contributions c⋅Δt, the weights w, the best bias z, the
        # residuals r and the best gain g.
        update_cΔt!(cΔt, c, Δt, cat, true)
        update_w!(w, cΔt, u)
        z = best_bias(w, d, cΔt)
        update_r!(r, d, cΔt, z)
        g = best_gain(w, r)

        # Compute the objective function.
        fx = zero(T)
        @inbounds @simd for i in 1:nframes
            # (5 ops + 1 log)/frames ~ 28 ops/frames
            fx += g*w[i]*r[i]^2 - log(w[i])
        end
        fx -= nframes*log(g)

        # Maybe update the best solution so far.
        if fx < res.f
            res.f = fx
            res.z = z
            res.g = g
            res.u = u
            copyto!(res.c, c)
        end

        # Compute the gradient of the objective function with respect to c.
        @inbounds for l in 1:ntypes
            gx[l] = zero(T)
        end
        @inbounds for i in 1:nframes
            # 8 ops/frames
            l = cat[i]
            gx[l] += w[i]*(1 - g*r[i]*(2 + w[i]*r[i]))*Δt[i]
        end

        # Compute the gradient of the objective function with respect to u.
        gu = zero(T)
        @inbounds @simd for i in 1:nframes
            # 6 ops/frames
            gu += w[i]*(1 - g*w[i]*r[i]^2)
        end
        gx[end] = gu

        # Return the objective function.
        return fx
    end

    # FIXME: stopping criterion
    vmlmb!(fg!, x, mem=5, lower=xmin)
end

"""

```julia
checkindices(I, len)
```

checks that all indices in `I` are in the range `1:len`.  An error is thrown
if `len ≤ 0` of if any values in `I` is outside the range `1:len`.

"""
function checkindices(I::AbstractArray{U}, len::Integer) where {U<:Unsigned}
    len > 0 || error("invalid length")
    if len < typemax(U)
        lim = U(len)
        @inbounds for i in I
            i - one(U) < lim || error("out of bound type index")
        end
    end
end

function checkindices(I::AbstractArray{S}, len::Integer) where {S<:Signed}
    len > 0 || error("invalid length")
    if len < typemax(S)
        U = unsigned(S)
        lim = U(len)
        @inbounds for i in I
            (i % U) - one(U) < lim || error("out of bound type index")
        end
    else
        # Just check for sign.
        @inbounds for i in I
            i > zero(U) || error("out of bound type index")
        end
    end
end

"""

```julia
update_w!(w, cΔt, u) -> w
```

overwrites `w` with `1/(c⋅Δt + u)`, that is do `∀i: w[i] = 1/(cΔt[i] + u)`, and
returns `w`.

See also [`update_cΔt!`](@ref).

"""
function update_w!(w::Vector{T}, cΔt::Vector{T}, u::T) where {T<:AbstractFloat}
    u′ = T(u)
    @inbounds @simd for i ∈ eachindex(w, cΔt)
       w[i] = one(T)/(cΔt[i] + u′)
    end
    return w
end

"""

```julia
update_cΔt!(cΔt, c, Δt, cat, nochecks=false) -> cΔt
```

overwrites `cΔt` with `c⋅Δt`, that is do `∀i: cΔt[i] = c[cat[i]]*Δt[i]`, and
returns `cΔt`.  Set optional argument `nochecks` to `true` to skip testing
the indices in `cat`.

See also  [`update_w!`](@ref), [`checkindices`](@ref).

"""
function update_cΔt!(cΔt::Vector{T}, c::Vector{T},
                     Δt::Vector{T}, cat::Vector{Int},
                     nochecks::Bool = false) where {T<:AbstractFloat}
    nframes = length(cΔt)
    @assert length(Δt) == nframes
    @assert length(cat) == nframes
    nochecks || checkindices(cat, length(c))
    @inbounds for i ∈ 1:nframes
        cΔt[i] = c[cat[i]]*Δt[i]
    end
    return cΔt
end


"""

```julia
update_r!(r, d, cΔt, z) -> r
```

overwrites array `r` with the residuals given the data `d`, the contribution
`cΔt` of the different sources and the bias `z`. The destination `r` is
returned.  The residuals are computed as:

```julia
∀i: r[i] = d[i] - cΔt[i] - z
```

See also [`update_cΔt!`](@ref), [`best_bias`](@ref).

"""
function update_r!(r::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T},
                   z::Real) where {T<:AbstractFloat}
    z′ = T(z)
    @inbounds @simd for i ∈ eachindex(r, d, cΔt)
        r[i] = d[i] - (cΔt[i] + z′)
    end
    return r
end

"""

```julia
best_bias(w, d, cΔt) -> z
```

yields the best bias given the weights `w`, the data `d` and the contribution
`cΔt` of the different sources.

See also [`update_w!`](@ref), [`update_cΔt!`](@ref), [`best_gain`](@ref).

"""
function best_bias(w::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T}) where {T<:AbstractFloat}
    a, b = zero(T), zero(T)
    @inbounds @simd for i ∈ eachindex(w, d, cΔt)
        a += w[i]
        b += w[i]*(d[i] - cΔt[i])
    end
    return b/a
end

"""

```julia
best_gain(w, d, cΔt, z) -> g
```

yields the best gain given the weights `w`, the data `d`, the contribution `cΔt`
of the different sources and the bias `z`.

Alternatively, if the residuals `r = d - cΔt - z` have been computed, just
call:

```julia
best_gain(w, r) -> g
```

See also [`update_w!`](@ref), [`update_cΔt!`](@ref), [`update_r!`](@ref),
[`best_bias`](@ref).

"""
function best_gain(w::AbstractVector{T},
                   d::AbstractVector{T},
                   cΔt::AbstractVector{T},
                   z::Real) where {T<:AbstractFloat}
    z′ = T(z)
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(w, d, cΔt)
        s += w[i]*(cΔt[i] + z′ - d[i])^2
    end
    return length(w)/s
end

function best_gain(w::AbstractVector{T},
                   r::AbstractVector{T}) where {T<:AbstractFloat}
    s = zero(T)
    @inbounds @simd for i ∈ eachindex(w, r)
        s += w[i]*r[i]^2
    end
    return length(w)/s
end

"""

```julia
leastpositive(A)
```

yields the least strictly positive value of array `A` or zero if all values of
`A` are nonpositive.

"""
function leastpositive(A::AbstractArray{T}) where {T}
    res = zero(T)
    @inbounds for val in A
        if val > zero(T) && (res > val || res == zero(T))
            res = val
        end
    end
    return res
end


#------------------------------------------------------------------------------

"""

```julia
calibrate(md, vd, ms, vs, mf = ms) -> cal
```

yields an instance of [`ReducedCalibration`](@ref) which
can be used to apply pre-processing of raw images acquired by a detector.  The
arguments are arrays of compatible sizes (see [`broadcast`](@ref)):

 * `md` and `vd` are the empirical mean and variance of a series of *dark*
   images that is raw images acquired with no illumination.

 * `ms` and `vs` are the empirical mean and variance of a series of raw images
   acquired with some *stable* illumination.

 * Optionally, `mf` is a mean raw *flat* image.  If not specified `ms` is used
   instead.

All images are assumed to have been acquired under the same conditions.  When
variances are needed (*i.e.*, for `vd` and `vs`), the corresponding series of
raw images must have been acquired under stable conditions (otherwise the
empirical variance also account for the variance of the instabilities).

Providing a *flat* image (different from `ms`) is meant to also compensate for
nonuniform transmission of the optics.  If `mf` is not supplied, `ms` should
correspond to a uniform illumination.

"""
function calibrate(md, vd, ms, vs, mf; kwds...)
    T = float(promote_type(eltype(md), eltype(vd),
                           eltype(ms), eltype(vs),
                           eltype(mf)))
    dims = bcastdims(size(md), size(vd),
                     size(ms), size(vs),
                     size(mf))
    return calibrate(bcastlazy(T, dims, md), bcastlazy(T, dims, vd),
                     bcastlazy(T, dims, ms), bcastlazy(T, dims, vs),
                     bcastlazy(T, dims, mf); kdws...)
end

function calibrate(md, vd, ms, vs; kwds...)
    T = float(promote_type(eltype(md), eltype(vd),
                           eltype(ms), eltype(vs)))
    dims = bcastdims(size(md), size(vd),
                     size(ms), size(vs))
    return calibrate(bcastlazy(T, dims, md), bcastlazy(T, dims, vd),
                     bcastlazy(T, dims, ms), bcastlazy(T, dims, vs); kdws...)
end

function calibrate(md::AbstractArray{T,N},
                   vd::AbstractArray{T,N},
                   ms::AbstractArray{T,N},
                   vs::AbstractArray{T,N},
                   mf::AbstractArray{T,N} = ms;
                   kwds...) where {T<:AbstractFloat,N}
    error("FIXME:")
    @assert !Base.has_offset_axes(md, vd, ms, vs, mf)
    dims = size(md)
    @assert size(vd) == dims
    @assert size(ms) == dims
    @assert size(vs) == dims
    @assert size(mf) == dims

    #FIXME: a = Array{T}(undef, dims)
    z = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    σ  = Array{T}(undef, dims)

    # The minimum variance, in (ADU/pixel/frame)^2, should be 1/12 which is the
    # variance of rounding to the nearest integer.  This is not used for now.
    minvar = zero(T)

    # Model of the flat distribution (FIXME: optionally fit a smooth
    # distribution).
    flt = one(T)

    # Default value for v to avoid division by zero.
    vdef = T(mean(flt))

    @inbounds for i ∈ eachindex(a, md, vd, ms, vs, mf)
        # a = flt/(mf - md)
        if isfinite(mf[i]) && isfinite(md[i]) && mf[i] > md[i]
            a[i] = flt/(mf[i] - md[i])
        else
            a[i] = 0
        end

        # z = md
        if isfinite(md[i])
            z[i] = md[i]
        else
            z[i] = 0
        end

        # σ = sqrt(vd)
        if isfinite(vd[i]) && vd[i] > minvar
            σ[i] = sqrt(vd[i])
        else
            σ[i] = 0
        end

        # g = (ms - md)/(vs - vd)
        if (isfinite(ms[i]) && isfinite(md[i]) && ms[i] > md[i] &&
            isfinite(vs[i]) && isfinite(vd[i]) && vs[i] > vd[i])
            g[i] = (ms[i] - md[i])/(vs[i] - vd[i])
        else
            g[i] = 0
        end

    end

    return ReducedCalibration(a, z, g, σ; kwds...)
end

end # module
