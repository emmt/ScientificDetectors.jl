
#------------------------------------------------------------------------------
# INPUT/OUTPUT

const WritableData{T,N} = Union{ReducedCalibration{T,N},
                                PreprocessingParameters{T,N}}

"""

```julia
write!(path, obj)
```

writes reduced detector calibration or preprocessing parameters `obj` in FITS
file `path`.

If the FITS file already exists, it is (silently) overwritten.  Call `write`
instead to throw an error if the file already exists.  The `write` method can
take a FITS handle instead of a file name, to append a new FITS HDU with
detector calibration parameters.

Call:

```julia
read(ReducedCalibration, src)
```

to read detector calibration parameters from source `src` (a file name or a
FITS handle).  Similarly, call:

```julia
read(PreprocessingParameters, src)
```

to read detector preprocessing parameters from source `src` (a file name or a
FITS handle).

"""
write!(path::AbstractString, obj::WritableData) =
    FitsIO(path, "w!") do io
        write(io, obj)
    end

function write(path::AbstractString, obj::WritableData;
               overwrite::Bool=false, kwds...)
    (overwrite == false && exists(path)) &&
        throw_file_already_exists(path, "call `write!` or use `overwrite=true`")
    FitsIO(path, (overwrite ? "w!" : "w")) do io
        write(io, obj)
    end
end

write(io::FitsIO, obj::WritableData; kwds...) =
    write(io, obj, FitsHeader(; kwds...))

function write(io::FitsIO, obj::ReducedCalibration{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 4 + length(obj.c))
    dat[…,1] .= obj.f
    dat[…,2] .= obj.z
    dat[…,3] .= obj.g
    dat[…,4] .= obj.σ
    k = 4
    for c in obj.c
        k += 1
        dat[…,k] .= c
    end

    # Create FITS header.
    name, vers = hduname(obj)
    hdr.HDUNAME  = (name, "reduced detector calibration")
    hdr.HDUVERS  = (vers, "version of this format")
    merge!(hdr, regionofinterest(obj))
    for k ∈ eachindex(obj.cat)
        hdr[string("CAT",k)] = obj.cat[k]
    end

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end

function write(io::FitsIO, obj::PreprocessingParameters{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 4)
    dat[…,1] .= obj.a
    dat[…,2] .= obj.b
    dat[…,3] .= obj.p
    dat[…,4] .= obj.q

    # Create FITS header.
    name, vers = hduname(obj)
    hdr.HDUNAME  = (name, "pre-processing parameters")
    hdr.HDUVERS  = (vers, "version of this format")
    hdr["EXPTIME"] = (exposuretime(obj), "[s] exposure time")
    merge!(hdr, regionofinterest(obj))

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end

# Extend EasyFITS method to provide HDU name and revision number.
hduname(::Type{<:ReducedCalibration}) = ("REDUCED-DETECTOR-CALIBRATION", 2)
hduname(::Type{<:PreprocessingParameters}) = ("DETECTOR-PREPROCESSING-PARAMETERS", 1)

function read(::Type{T}, path::AbstractString) where {T<:WritableData}
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
    version = read(Int, hdu, "HDUVERS", 0)
    # FIXME: automatically convert rev 1 into rev 2
    version == 2 || error(string("unsupported format revision ", version))
    return read(T, hdu)
end

function read(::Type{ReducedCalibration{T,N}},
              hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    length(size(hdu)) == N+1 || dimension_mismatch("invalid number of dimensions")
    return read(ReducedCalibration{T}, hdu)
end

function read(::Type{T}, hdu::FitsImageHDU) where {T<:ReducedCalibration}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, vers = hduname(T)
    cname = read(String, hdu, "HDUNAME", "")
    cname == name || error(string("unexpected HDUNAME \"", cname, "\""))
    cvers = read(Int, hdu, "HDUVERS", 0)
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
    roi = get(NTuple{N,DetectorAxis}, hdr)
    cat = Vector{String}(undef, nc)
    for k in 1:nc
        cat[k] = get(hdr, string("CAT",k), "")
    end

    # Read data and build instance.
    colons = rubberindex(N)
    f = read(hdu, colons..., 1)
    z = read(hdu, colons..., 2)
    g = read(hdu, colons..., 3)
    σ = read(hdu, colons..., 4)
    c = Vector{typeof(f)}(undef, nc)
    for k in 1:nc
        c[k] = read(hdu, colons..., 4 + k)
    end
    return T(roi, f, z, g, σ, c, cat)
end

function read(::Type{T}, io::FitsIO) where {T<:PreprocessingParameters}
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(hdu -> read(String, hdu, "HDUNAME", nothing) == name, io)
    k === nothing && error("no reduced detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    version = read(Int, hdu, "HDUVERS", 0)
    version == 1 || error(string("unsupported format revision ", version))
    return read(T, hdu)
end

function read(::Type{PreprocessingParameters{T,N}},
              hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    length(size(hdu)) == N+1 || dimension_mismatch("invalid number of dimensions")
    return read(PreprocessingParameters{T}, hdu)
end

function read(::Type{T}, hdu::FitsImageHDU) where {T<:PreprocessingParameters}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, vers = hduname(T)
    cname = read(String, hdu, "HDUNAME", "")
    cname == name || error(string("unexpected HDUNAME \"", cname, "\""))
    cvers = read(Int, hdu, "HDUVERS", 0)
    cvers == vers || error(string("unsupported format revision ", cvers))
    bitpix = read(Int, hdu, "BITPIX")
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    dims = size(hdu)
    N = length(dims) - 1
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims[end] == 4 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = read(FitsHeader, hdu)
    roi = get(NTuple{N,DetectorAxis}, hdr)
    Δt = hdr["EXPTIME"]

    # Read data and build instance.
    colons = rubberindex(N)
    a = read(hdu, colons..., 1)
    b = read(hdu, colons..., 2)
    p = read(hdu, colons..., 3)
    q = read(hdu, colons..., 4)
    return T(roi, Δt, a, b, p, q)
end
