#
# io.jl -
#
# Input/output methods for loading/saving calibration and preprocessing
# parameters from/to files.
#

const WritableData{T,N} = Union{PreprocessingParameters{T,N},
                                ReducedCalibration{T,N},
                                SampleStatistics{T,N},
                                SimpleCalibration{T,N}}

"""
    write!(path, obj)

writes reduced detector calibration or preprocessing parameters `obj` in FITS
file `path`.

If the FITS file already exists, it is (silently) overwritten.  Call `write`
instead to throw an error if the file already exists.  The `write` method can
take a FITS handle instead of a file name, to append a new FITS HDU with
detector calibration parameters.

Call:

    read(ReducedCalibration, src)

to read detector calibration parameters from source `src` (a file name or a FITS
handle).  Similarly, call:

    read(PreprocessingParameters, src)

to read detector preprocessing parameters from source `src` (a file name or a
FITS handle).

"""
write!(path::AbstractString, obj::WritableData; kwds...) =
    writefits!(path, obj; kwds...)

write(path::AbstractString, obj::WritableData; kwds...) =
    writefits(path, obj; kwds...)

read(::Type{T}, path::AbstractString) where {T<:WritableData} =
    readfits(T, path)

writefits!(path::AbstractString, obj::WritableData; kwds...) =
    writefits(path, obj; overwrite=true, kwds...)

function writefits(path::AbstractString, obj::WritableData;
                   overwrite::Bool=false, kwds...)
    (overwrite == false && ispath(path)) &&
        throw_file_already_exists(path, "call `write!` or use `overwrite=true`")
    FitsFile(path, (overwrite ? "w!" : "w")) do io
        write(io, obj; kwds...)
    end
end

write(io::FitsFile, obj::WritableData; kwds...) =
    write(io, obj, FitsHeader(; kwds...))

function readfits(::Type{T}, path::AbstractString) where {T<:WritableData}
    FitsFile(path, "r") do io
        return read(T, io)
    end
end

#------------------------------------------------------------------------------
#
# I/O methods for `ReducedCalibration`.
#

# Extend EasyFITS method to provide HDU name and revision number.
hduname(::Type{<:ReducedCalibration}) = ("REDUCED-DETECTOR-CALIBRATION", 3)

function read(R::Type{<:ReducedCalibration}, io::FitsFile)
    # Find HDU with calibration parameters.
    name, vers = hduname(R)
    k = findfirst(H -> matchvalue(H, "HDUNAME", name), io)
    k === nothing && error("no reduced detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    return read(R, hdu)
end

function read(::Type{ReducedCalibration}, hdu::FitsImageHDU{T}) where {T}
   return read(ReducedCalibration{float(T)}, hdu)
end

function read(::Type{ReducedCalibration{T,N}},
              hdu::FitsImageHDU{<:Any,Np1}) where {T<:AbstractFloat,N,Np1}
    Np1 == N + 1 || dimension_mismatch("incompatible number of dimensions")
    return read(ReducedCalibration{T}, hdu)
end

function read(::Type{ReducedCalibration{T}},
              hdu::FitsImageHDU{<:Any,N}) where {T<:AbstractFloat,N}
    # Check HDUNAME, HDUVERS, and BITPIX.
    name, _ = hduname(ReducedCalibration)
    matchvalue(hdu, "HDUNAME", name) || error("invalid HDUNAME")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")
    1 ≤ version ≤ 3 || error("unsupported format revision $version")

    # Check dimensions.
    dims = hdu.data_size
    @assert length(dims) == N
    N > 1 || dimension_mismatch("invalid number of dimensions")
    n1 = (version < 3 ? 4 : 5) # number of fields before the sources
    nsrc = dims[end] - n1
    nsrc ≥ 0 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = get(NTuple{N - 1, DetectorAxis}, hdr)
    src = [getvalue(String, hdr, "SRC$k", "") for k in 1:nsrc]

    # Read data and build instance.
    inds = colons(N - 1)
    f = read(hdu, inds..., 1)
    z = read(hdu, inds..., 2)
    g = read(hdu, inds..., 3)
    σ = read(hdu, inds..., 4)
    bpm = n1 ≥ 5 ? read(hdu, inds..., 5) : FastUniformArray(true, dims[1:end-1])
    s = Vector{typeof(f)}(undef, nsrc)
    for k in 1:nsrc
        s[k] = read(hdu, inds..., n1 + k)
    end
    return ReducedCalibration{T}(roi, f, z, g, σ, s, src; bpm=bpm)
end

function write(io::FitsFile, obj::ReducedCalibration{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 5 + length(obj.s))
    dat[..,1] .= obj.f
    dat[..,2] .= obj.z
    dat[..,3] .= obj.g
    dat[..,4] .= obj.σ
    dat[..,5] .= obj.bpm
    k = 5
    for s in obj.s
        k += 1
        dat[..,k] .= s
    end

    # Create FITS header.
    name, vers = hduname(obj)
    hdr["HDUNAME"] = (name, "reduced detector calibration")
    hdr["HDUVERS"] = (vers, "version of this format")
    merge!(hdr, DetectorAxes(obj))
    for k ∈ eachindex(obj.src)
        hdr[string("SRC",k)] = obj.src[k]
    end

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end

#------------------------------------------------------------------------------
#
# I/O methods for `SimpleCalibration`.
#

# Extend EasyFITS method to provide HDU name and revision number.
hduname(::Type{<:SimpleCalibration}) = ("SIMPLE-DETECTOR-CALIBRATION", 1)

function read(::Type{T}, io::FitsFile) where {T<:SimpleCalibration}
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(H -> matchvalue(H, "HDUNAME", name), io)
    k === nothing && error("no simple detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    version == 1 || error(string("unsupported format revision ", version))
    return read(T, hdu)
end

function read(::Type{SimpleCalibration{T,N}},
              hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    dims = hdu.data_size
    length(dims) == N+1 || dimension_mismatch("invalid number of dimensions")
    dims[end] == 5 || dimension_mismatch("invalid last dimension")
    return read(SimpleCalibration{T}, hdu)
end

function read(::Type{T}, hdu::FitsImageHDU) where {T<:SimpleCalibration}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, vers = hduname(T)
    cname = getvalue(String, hdu, "HDUNAME", "")
    cname == name || error(string("unexpected HDUNAME \"", cname, "\""))
    cvers = getvalue(Int, hdu, "HDUVERS", 0)
    cvers == vers || error(string("unsupported format revision ", cvers))
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    dims = hdu.data_size
    N = length(dims) - 1
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims[end] == 5 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = get(NTuple{N,DetectorAxis}, hdr)
    Δt = hdr["EXPTIME"].float

    # Read data and build instance.
    inds = colons(N)
    f = read(hdu, inds..., 1)
    a = read(hdu, inds..., 2)
    b = read(hdu, inds..., 3)
    g = read(hdu, inds..., 4)
    σ = read(hdu, inds..., 5)
    return T(roi, Δt, f, a, b, g, σ)
end

function write(io::FitsFile, obj::SimpleCalibration{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 5)
    dat[..,1] .= obj.f
    dat[..,2] .= obj.a
    dat[..,3] .= obj.b
    dat[..,4] .= obj.g
    dat[..,5] .= obj.σ

    # Create FITS header.
    name, vers = hduname(obj)
    hdr["HDUNAME"] = (name, "simple detector calibration")
    hdr["HDUVERS"] = (vers, "version of this format")
    hdr["EXPTIME"] = (exposuretime(obj), "[s] exposure time")
    merge!(hdr, DetectorAxes(obj))

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end


#------------------------------------------------------------------------------
#
# I/O methods for `PreprocessingParameters`.
#

# Extend EasyFITS method to provide HDU name and revision number.
hduname(::Type{<:PreprocessingParameters}) =
    ("DETECTOR-PREPROCESSING-PARAMETERS", 1)

function read(::Type{T}, io::FitsFile) where {T<:PreprocessingParameters}
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(hdu -> matchvalue(hdu, "HDUNAME", name), io)
    k === nothing && error("no detector pre-processing parameters found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    version == 1 || error(string("unsupported format revision ", version))
    return read(T, hdu)
end

function read(::Type{PreprocessingParameters{T,N}},
              hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    dims = hdu.data_size
    length(dims) == N+1 || dimension_mismatch("invalid number of dimensions")
    dims[end] == 4 || dimension_mismatch("invalid last dimension")
    return read(PreprocessingParameters{T}, hdu)
end

function read(::Type{T}, hdu::FitsImageHDU) where {T<:PreprocessingParameters}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, vers = hduname(T)
    cname = getvalue(String, hdu, "HDUNAME", "")
    cname == name || error(string("unexpected HDUNAME \"", cname, "\""))
    cvers = getvalue(Int, hdu, "HDUVERS", 0)
    cvers == vers || error(string("unsupported format revision ", cvers))
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    dims = hdu.data_size
    N = length(dims) - 1
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims[end] == 4 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = get(NTuple{N,DetectorAxis}, hdr)
    Δt = hdr["EXPTIME"].float

    # Read data and build instance.
    inds = colons(N)
    a = read(hdu, inds..., 1)
    b = read(hdu, inds..., 2)
    q = read(hdu, inds..., 3)
    r = read(hdu, inds..., 4)
    return T(roi, Δt, a, b, q, r)
end

function write(io::FitsFile, obj::PreprocessingParameters{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 4)
    dat[..,1] .= obj.a
    dat[..,2] .= obj.b
    dat[..,3] .= obj.q
    dat[..,4] .= obj.r

    # Create FITS header.
    name, vers = hduname(obj)
    hdr["HDUNAME"] = (name, "pre-processing parameters")
    hdr["HDUVERS"] = (vers, "version of this format")
    hdr["EXPTIME"] = (exposuretime(obj), "[s] exposure time")
    merge!(hdr, DetectorAxes(obj))

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end

#------------------------------------------------------------------------------
#
# I/O methods for `SampleStatistics`.
#

hduname(::Type{<:SampleStatistics}) = ("DETECTOR-SAMPLE-STATISTICS", 1)

"""
# Sample Statistics

There are 2 generations of FITS file with image statistics prior to the
specification HDUNAME="DETECTOR-STATISTICS" (rev. 1). The different headers are
summarized below.

| Keyword  | Type    | Description                                     |
|:---------|:--------|:------------------------------------------------|
| XOFFSET  | Integer | Offset of region of interest along first axis   |
| YOFFSET  | Integer | Offset of region of interest along second axis  |
| DEPTH    | Integer | Bits per pixel                                  |
| GAIN     | Integer | Detector gain                                   |
| BIAS     | Integer | Detector black level                            |
| EXPOSURE | Integer | Exposure time [microsec]                        |
| RATE     | Integer | Frames per second [Hz]                          |
| SAMPLES  | Integer | Number of averaged images                       |

| Keyword  | Type    | Description                                     |
|:---------|:--------|:------------------------------------------------|
| XBIN     | Integer | Pixel binning factor along first axis           |
| YBIN     | Integer | Pixel binning factor along second axis          |
| XOFFSET  | Integer | Offset of region of interest along first axis   |
| YOFFSET  | Integer | Offset of region of interest along second axis  |
| EXPOSURE | Real    | Exposure time [seconds]                         |
| RATE     | Integer | Frames per second [Hz]                          |
| SAMPLES  | Integer | Number of averaged images                       |

| Keyword  | Type    | Value/Description                               |
|:---------|:--------|:------------------------------------------------|
| HDUNAME  | String  | "DETECTOR-SAMPLE-STATISTICS"                    |
| HDUVERS  | Integer | 1                                               |
| BIN1     | Integer | Pixel binning factor along axis 1               |
| BIN2     | Integer | Pixel binning factor along axis 2               |
| OFF1     | Integer | Offset of region of interest along axis 1       |
| OFF2     | Integer | Offset of region of interest along axis 2       |
| EXPTIME  | Real    | [s] Exposure time                               |
| SAMPLES  | Integer | Number of averaged images                       |
| CATEGORY | String  | Type of illumination source                     |

See https://heasarc.gsfc.nasa.gov/docs/fcg/common_dict.html for a list of
commonly used FITS keywords.

The array data, say `arr`, stored in the FITS HDU (header data unit) has
last dimension equal to 2 and consists in 2 packed arrays `arr[..,1]` and
`arr[..,2]` which are respectively the mean and standard deviation of the
sample.   These quantities are computed as follows:

    arr[i,1] = (dat1[i] + dat2[i] + ... + datn[i])/n
    arr[i,2] = sqrt(((dat1[i] - arr[i,1])^2 + ... +
                    (datn[i] - arr[i,1])^2)/(n - 1))

with `i` the multi-dimensional index, `n` the number of data samples,
`dat1` the first data sample, ..., and `datn` the last data sample.  Note
that the square of the standard deviation gives an unbiased estimator of
the variance.

""" _read1

function read(::Type{T}, io::FitsFile) where {T<:SampleStatistics}
    for i in 1:length(io)
        hdu = io[i]
        tup = _read1(T, hdu)
        if tup !== nothing
            return _read2(T, hdu, tup...)
        end
    end
    error("no detector sample statistics found")
end

function read(::Type{SampleStatistics{T,N}},
              hdu::FitsHDU) where {T<:AbstractFloat,N}
    tup = _read1(SampleStatistics, hdu)
    tup === nothing && error("HDU does not contain detector sample statistics")
    length(tup[3]) == N || dimension_mismatch("invalid number of dimensions")
    return _read2(SampleStatistics{T}, hdu, tup...)
end

function read(::Type{T}, hdu::FitsHDU) where {T<:SampleStatistics}
    tup = _read1(SampleStatistics, hdu)
    tup === nothing && error("HDU does not contain detector sample statistics")
    return _read2(T, hdu, tup...)
end

function _read1(::Type{T}, hdu::FitsHDU) where{T<:SampleStatistics}
    # First try to extract information from new format.
    name, lastvers = hduname(T)
    if matchvalue(hdu, "HDUNAME", name)
        # New format.
        hdu isa FitsImageHDU || error("expecting FITS Image HDU")
        thisvers = getvalue(hdu, "HDUVERS", 0)
        thisvers == lastvers || error(
            "unsupported version = $thisvers for HDUNAME = \"$name\"")
        exptime = hdu["EXPTIME"].float
        samples = hdu["SAMPLES"].integer
        dims = hdu.data_size
        if length(dims) < 2
            error("invalid number of dimensions for sample statistics")
        elseif dims[end] != 2
            error("invalid last dimension for sample statistics")
        end
        N = length(dims) - 1
        off = Vector{Int}(undef, N)
        bin = Vector{Int}(undef, N)
        for d in 1:N
            off[d] = hdu["OFF$d"].integer
            bin[d] = hdu["BIN$d"].integer
        end
        roi = ntuple(i -> DetectorAxis(dims[i]; off = off[i], bin = bin[i]), N)
        return samples, exptime, roi
    end

    # Maybe old format, only in primary HDU.
    hdu.number == 1 || return nothing
    (samples = getvalue(Int, hdu, "SAMPLES", nothing)) === nothing && return nothing
    (exposure = getvalue(hdu, "EXPOSURE", nothing)) === nothing && return nothing
    (xoff = getvalue(Int, hdu, "XOFFSET", nothing)) === nothing && return nothing
    (yoff = getvalue(Int, hdu, "YOFFSET", nothing)) === nothing && return nothing
    xbin = getvalue(Int, hdu, "XBIN", nothing)
    ybin = getvalue(Int, hdu, "YBIN", nothing)
    dims = hdu.data_size
    mesg = (
        length(dims) != 3 ?
        "invalid number of dimensions for sample statistics" :
        dims[end] != 2 ?
        "invalid last dimension for sample statistics" : "")
    if exposure isa Integer && xbin === nothing && ybin === nothing
        # Assume exposure time in microseconds.
        mesg == "" || error(mesg)
        exptime = Float64(exposure*1e-6)
        roi = (DetectorAxis(dims[1]; off = xoff, bin = 1),
               DetectorAxis(dims[2]; off = yoff, bin = 1))
        return samples, exptime, roi
    end
    if exposure isa AbstractFloat && xbin isa Int && ybin isa Int
        # Assume exposure time in seconds.
        mesg == "" || error(mesg)
        exptime = Float64(exposure)
        roi = (DetectorAxis(dims[1]; off = xoff, bin = xbin),
               DetectorAxis(dims[2]; off = yoff, bin = ybin))
        return samples, exptime, roi
    end
    return nothing
end

function _read2(::Type{T}, hdu::FitsImageHDU,
                samples::Integer, Δt::Float64,
                roi::NTuple{N,DetectorAxis}) where {T<:SampleStatistics,N}
    # Read data and build instance.
    inds = colons(N)
    avg = read(hdu, inds..., 1)
    std = read(hdu, inds..., 2)
    return T(avg, std, samples, Δt, roi)
end

function write(io::FitsFile, obj::SampleStatistics{T,N},
               hdr::FitsHeader) where {T,N}
    # Create data array.
    dims = size(obj)
    dat = Array{T,N+1}(undef, dims..., 2)
    dat[..,1] .= mean(obj)
    dat[..,2] .= std(obj)

    # Create FITS header.
    name, vers = hduname(obj)
    hdr["HDUNAME"] = (name, "detector sample statistics")
    hdr["HDUVERS"] = (vers, "version of this format")
    hdr["EXPTIME"] = (exposuretime(obj), "[s] exposure time")
    hdr["SAMPLES"] = (nobs(obj), "number of samples")
    merge!(hdr, DetectorAxes(obj))

    # Write FITS HDU.
    write(io, dat, hdr)
    nothing
end
