module Calibration

import Base: read, write

using Statistics, Printf
using ArrayTools

using EasyFITS
using EasyFITS: exists, throw_file_already_exists
import EasyFITS: write!, hduname

# FIXME: add fitting of the smooth model

const Colons{N} = NTuple{N,Colon}

"""

`ScientificDetectors.ReducedCalibration{T}` stores the calibration
parameters with `T` the floating-point type for the computations.

Constructor is called as:

```julia
ScientificDetectors.ReducedCalibration([T,] a, z, g, s, args...; kwds...) -> cal
```

where `T` is the floating-point type for the computations (optional, if
unspecified, it is deduced from the element type of the other arguments), `a`
is the amplitude correction factor (in flux units per ADU), `z` is the *zero
level* that is the constant bias set by the analog to digital converter (in
ADU), `g` is the detector gain (in electrons per ADU) and `s` is the standard
deviation of the readout noise (in ADU/frame).  Arguments `a`, `z`, `g` and `s`
are pixelwise, they are broadcast to common dimensions (which should be that of
the detector) and their elements converted to the same type `T`.

Additional arguments `args...` are key-value pairs like `"id1"=>c1`,
`:id2=>c2`, ... of identifiers and arrays corresponding to current terms
like the dark current or any background flux (in ADU/second).  Arguments
`c1`, `c2`, ... are assumed to be pixelwise.

Keywords `xoff` and `yoff` (both `0` by default) can be used to specify the
horizontal and vertical offsets (in physical pixels) of the region of
interest (ROI) that has been calibrated.

Keywords `xbin` and `ybin` (both `1` by default) can be used to specify the
horizontal and vertical binning factors (in physical pixels).

Basic operations on `ScientificDetectors.ReducedCalibration` instance `obj`:

```julia
size(obj)   # yields the dimensions of the detector
size(obj,k) # yields the `k`-th dimension of the detector
length(obj) # yields the number of elements of the detector
eltype(obj) # yields the floating-point type of the calibration data
T.(obj)     # convert contents of `obj` to floating-point type `T`
```

"""
struct ReducedCalibration{T<:AbstractFloat,N}
    # Dimensions.
    dims::NTuple{N,Int}

    # Offsets of the calibrated ROI with respect to the sensor.
    xoff::Int
    yoff::Int

    # Dimensions of the macro-pixels (in physical pixels).
    xbin::Int
    ybin::Int

    # Amplitude correction factor (in flux units per ADU):
    a::Array{T,N}

    # Zero-level (constant bias in ADU):
    z::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise (in ADU/frame):
    s::Array{T,N}

    # Time dependent bias, e.g. dark current and background flux, (in
    # ADU/second), may be empty or zero-filled:
    c::Vector{Array{T,N}}

    # Identifiers of the calibration sources responsible of the different
    # time-dependent bias terms.
    cids::Vector{String}

    # Inner constructor provided to force using outer constructors.
    function ReducedCalibration{T,N}(dims::NTuple{N,Int},
                                     xoff::Integer,
                                     yoff::Integer,
                                     xbin::Integer,
                                     ybin::Integer,
                                     a::Array{T,N},
                                     z::Array{T,N},
                                     g::Array{T,N},
                                     s::Array{T,N},
                                     c::Vector{Array{T,N}},
                                     cids::Vector{String}
                                     ) where {T<:AbstractFloat,N}
        @assert xoff ≥ 0
        @assert yoff ≥ 0
        @assert xbin ≥ 1
        @assert ybin ≥ 1
        @assert size(a) == dims
        @assert size(z) == dims
        @assert size(g) == dims
        @assert size(s) == dims
        @assert length(cids) == length(c)
        for k in eachindex(c)
            @assert size(c[k]) == dims
            all(x -> isfinite(x) && x ≥ 0, c[k]) ||
                error("some invalid values in time-dependent bias")
        end
        all(x -> isfinite(x) && x ≥ 0, a) ||
            error("some invalid values in amplitude correction")
        all(x -> isfinite(x), z) ||
            error("some invalid values in constant bias")
        all(x -> isfinite(x) && x ≥ 0, g) ||
            error("some invalid values in detector gain")
        all(x -> isfinite(x) && x ≥ 0, s) ||
            error("some invalid values in readout noise")
        return new{T,N}(dims, xoff, yoff, xbin, ybin, a, z, g, s, c, cids)
    end
end

#
# Outer constructors for ReducedCalibration structure.
#

# A constructor of an immutable structure can return its argument.
ReducedCalibration(obj::ReducedCalibration) = obj
ReducedCalibration{T}(obj::ReducedCalibration{T}) where {T} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{T,N}) where {T,N} = obj
ReducedCalibration{T,N}(obj::ReducedCalibration{<:Any,N}) where {T,N} =
    ReducedCalibration{T}(obj)
ReducedCalibration{T}(obj::ReducedCalibration{<:Any,N}) where {T<:AbstractFloat,N} =
    ReducedCalibration{T,N}(obj.dims, obj.xoff, obj.yoff, obj.xbin, obj.ybin,
                            convert(Array{T,N}, obj.a),
                            convert(Array{T,N}, obj.z),
                            convert(Array{T,N}, obj.g),
                            convert(Array{T,N}, obj.s),
                            map(x -> convert(Array{T,N}, x), obj.c),
                            obj.cids)

function ReducedCalibration(a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N};
                            kwds...) where {N}
    return ReducedCalibration(promote_eltype(a, z, g, s), a, z, g, s; kwds...)
end

function ReducedCalibration(::Type{T},
                            a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N};
                            kwds...) where {T,N}
    return ReducedCalibration(T, a, z, g, s, Array{T,N}[], String[]; kwds...)
end

function ReducedCalibration(a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N},
                            args...;
                            kwds...) where {N}
    return ReducedCalibration(a, z, g, s, _getc(args...)...; kwds...)
end

function ReducedCalibration(::Type{T},
                            a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N},
                            args...; kwds...) where {T,N}
    return ReducedCalibration(T, a, z, g, s, _getc(args...)...; kwds...)
end

# Convert pairs like "key1"=>arr1, :key2=>arr2, ... in a list of
# arrays and a list of identifiers.
_getc(args::Pair{<:Union{AbstractString,Symbol},<:AbstractArray}...) =
    (collect(map(x -> x[2], args)),
     collect(map(x -> _string(x[1]), args)))

# Convert argument to a string as fast as possible.
_string(x::String) = x
_string(x::AbstractArray) = String(x)
_string(x::Symbol) = String(x)
@noinline _string(::T) where {T} =
    throw(ArgumentError(string("cannot convert argument of type `", T,
                               "` into a string")))

function ReducedCalibration(a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N},
                            c::AbstractVector{<:AbstractArray{<:Any,N}},
                            cids::AbstractVector;
                            kwds...) where {N}
    T = _promote_eltype(promote_eltype(a, z, g, s), c, length(c))
    return ReducedCalibration(T, a, z, g, s, c, cids; kwds...)
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

function ReducedCalibration(::Type{T},
                            a::AbstractArray{<:Any,N},
                            z::AbstractArray{<:Any,N},
                            g::AbstractArray{<:Any,N},
                            s::AbstractArray{<:Any,N},
                            c::AbstractVector{<:AbstractArray{<:Any,N}},
                            cids::AbstractVector;
                            kwds...) where {T<:AbstractFloat,N}
    T <: AbstractFloat ||
        error("promoted element types must be floating-point")
    return ReducedCalibration(T,
                              convert(Array{T,N}, a),
                              convert(Array{T,N}, z),
                              convert(Array{T,N}, g),
                              convert(Array{T,N}, s),
                              map(x -> convert(Array{T,N}, x), c),
                              map(x -> convert(String, x), cids);
                              kwds...)
end

function ReducedCalibration(::Type{T},
                            a::Array{T,N},
                            z::Array{T,N},
                            g::Array{T,N},
                            s::Array{T,N},
                            c::Vector{Array{T,N}},
                            cids::Vector{String};
                            xoff::Integer = 0,
                            yoff::Integer = 0,
                            xbin::Integer = 1,
                            ybin::Integer = 1) where {T<:AbstractFloat,N}
    return ReducedCalibration{T,N}(size(a), xoff, yoff, xbin, ybin,
                                   a, z, g, s, c, cids)
end

#
# Basic operations on ReducedCalibration structure.
#
Base.eltype(::ReducedCalibration{T}) where {T} = T
Base.size(obj::ReducedCalibration) = obj.dims
Base.size(obj::ReducedCalibration, k) = obj.dims[k]
Base.length(obj::ReducedCalibration) = prod(size(obj))
Base.convert(::Type{ReducedCalibration}, obj::ReducedCalibration) = obj
Base.convert(::Type{ReducedCalibration{T}}, obj::ReducedCalibration) where {T<:AbstractFloat} = T.(obj)

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::ReducedCalibration{T}) where {T} = obj
Broadcast.broadcasted(::Type{T}, obj::ReducedCalibration) where {T<:AbstractFloat} =
    ReducedCalibration{T}(obj)

"""

```julia
ScientificDetectors.calibrate(md, vd, ms, vs, mf = ms) -> cal
```

yields an instance of [`ScientificDetectors.ReducedCalibration`](@ref) which
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
    @assert !Base.has_offset_axes(md, vd, ms, vs, mf)
    dims = size(md)
    @assert size(vd) == dims
    @assert size(ms) == dims
    @assert size(vs) == dims
    @assert size(mf) == dims

    a = Array{T}(undef, dims)
    z = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    s  = Array{T}(undef, dims)

    # The minimum variance, in (ADU/pixel/frame)^2, should be 1/12 which is the
    # variance of rounding to the nearest integer.  This is not used for now.
    minvar = zero(T)

    # Model of the flat distribution (FIXME: optionally fit a smooth
    # distribution).
    flt = one(T)

    # Default value for v to avoid division by zero.
    vdef = T(mean(flt))

    @inbounds for i in eachindex(a, md, vd, ms, vs, mf)
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

        # s = sqrt(vd)
        if isfinite(vd[i]) && vd[i] > minvar
            s[i] = sqrt(vd[i])
        else
            s[i] = 0
        end

        # g = (ms - md)/(vs - vd)
        if (isfinite(ms[i]) && isfinite(md[i]) && ms[i] > md[i] &&
            isfinite(vs[i]) && isfinite(vd[i]) && vs[i] > vd[i])
            g[i] = (ms[i] - md[i])/(vs[i] - vd[i])
        else
            g[i] = 0
        end

    end

    return ReducedCalibration(a, z, g, s; kwds...)
end

"""

```julia
write!(path, calib)
```

writes detector calibration parameters `calib` in FITS file `path`.

If the FITS file already exists, it is (silently) overwritten.  Call `write`
instead to throw an error if the file already exists.  The `write` method can
take a FITS handle instead of a file name, to append a new FITS HDU with
detector calibration parameters.

Call

```julia
read(ScientificDetectors.ReducedCalibration, src) -> calib
```

to read detector calibration parameters from source `src` (a file name or a
FITS handle).

"""
write!(path::AbstractString, calib::ReducedCalibration) =
    FitsIO(path, "w!") do io
        write(io, calib)
    end

function write(path::AbstractString, calib::ReducedCalibration;
               overwrite::Bool=false, kwds...)
    (overwrite == false && exists(path)) &&
        throw_file_already_exists(path, "call `write!` or use `overwrite=true`")
    FitsIO(path, (overwrite ? "w!" : "w")) do io
        write(io, calib)
    end
end

write(io::FitsIO, calib::ReducedCalibration{T,N}; kwds...) =
    write(io, calib, FitsHeader(; kwds...))

function write(io::FitsIO, calib::ReducedCalibration{T,N},
               hdr::FitsHeader) where {T,N}

    # Create data array.
    dims = size(calib)
    dat = Array{T,N+1}(undef, dims..., 4 + length(calib.c))
    dat[…,1] .= calib.a
    dat[…,2] .= calib.z
    dat[…,3] .= calib.g
    dat[…,4] .= calib.s
    k = 4
    for c in calib.c
        k += 1
        dat[…,k] .= c
    end

    # Create FITS header.
    name, vers = hduname(calib)
    hdr.HDUNAME  = (name, "Reduced detector calibration")
    hdr.HDUVERS  = (vers, "Version of this format")
    hdr.XOFFSET  = (calib.xoff, "Horizontal offset (in physical pixels)")
    hdr.YOFFSET  = (calib.yoff, "Vertical offset (in physical pixels)")
    hdr.XBINNING = (calib.xbin, "Horizontal binning (in physical pixels)")
    hdr.YBINNING = (calib.ybin, "Vertical binning (in physical pixels)")
    for k in eachindex(calib.cids)
        hdr[string("CALIB",k)] = calib.cids[k]
    end

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end

# Extend EasyFITS method to provide HDU name and revision number.
hduname(::Type{<:ReducedCalibration}) = ("REDUCED-DETECTOR-CALIBRATION", 1)

function read(::Type{T}, path::AbstractString) where {T<:ReducedCalibration}
    FitsIO(path, "r") do io
        return read(T, io)
    end
end

function read(::Type{T}, io::FitsIO) where {T<:ReducedCalibration}
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(hdu -> read(String, hdu, "HDUNAME", nothing) == name, io)
    k === nothing && error("no reduced detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    version = read(Int, hdu, "HDUVERS", 1)
    version == 1 || error(string("unsupported format revision ", version))
    return read(T, hdu)
end

function read(::Type{ReducedCalibration{T,N}}, hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    length(size(hdu)) == N+1 || dimension_mismatch("invalid number of dimensions")
    return convert(ReducedCalibration{T,N}, read(ReducedCalibration, hdu))
end

function read(::Type{T}, hdu::FitsImageHDU) where {T<:ReducedCalibration}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, vers = hduname(T)
    cname = read(String, hdu, "HDUNAME", "")
    cname == name || error(string("unexpected HDUNAME \"", cname, "\""))
    cvers = read(Int, hdu, "HDUVERS", 1)
    cvers == vers || error(string("unsupported format revision ", cvers))
    bitpix = read(Int, hdu, "BITPIX")
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    dims = size(hdu)
    N = length(dims) - 1
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    nc = dims[end] - 4
    nc ≥ 0 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = read(FitsHeader, hdu)
    kwds = (
        xoff = get(hdr, "XOFFSET",  0),
        yoff = get(hdr, "YOFFSET",  0),
        xbin = get(hdr, "XBINNING", 1),
        ybin = get(hdr, "YBINNING", 1),
    )
    cids = Vector{String}(undef, nc)
    for k in 1:nc
        cids[k] = get(hdr, string("CALIB",k), "")
    end

    # Read data and build instance.
    colons = rubberindex(N)
    a = read(hdu, colons..., 1)
    z = read(hdu, colons..., 2)
    g = read(hdu, colons..., 3)
    s = read(hdu, colons..., 4)
    c = Vector{typeof(a)}(undef, nc)
    for k in 1:nc
        c[k] = read(hdu, colons..., 4 + k)
    end
    return convert(T, ReducedCalibration(a, z, g, s, c, cids; kwds...))
end

@noinline dimension_mismatch() =
    dimension_mismatch("arguments have incompatible dimensions")

@noinline dimension_mismatch(mesg::AbstractString) =
    throw(DimensionMismatch(mesg))

end # module
