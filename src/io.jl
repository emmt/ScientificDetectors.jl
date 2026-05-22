#
# io.jl -
#
# Input/output methods for loading/saving calibration and preprocessing
# parameters from/to files.
#

const WritableData{T,N} = Union{PreprocessingParameters{T,N},
                                CalibrationData{T,N},
                                ReducedCalibration{T,N},
                                SampleStatistics{T,N},
                                SimpleCalibration{T,N}}

# HDU name and revision number only depend on the type of our objects..
AstroFITS.hduname(data::WritableData) = hduname(typeof(data))

@deprecate(Base.write(filename::AbstractString, hdr, data::WritableData; overwrite::Bool = false),
    writefits(filename, hdr, data; overwrite = overwrite),false)
@deprecate(Base.write(filename::AbstractString, data::WritableData; kwds...),
    writefits(filename, data; kwds...), false)

"""
    writefits(dest, [hdr,] data; kwds...)

Write reduced detector calibration or preprocessing parameters `data` in `dest` which may
be a FITS file instance or the name of a new FITS file to create. Argument `hdr` is an
optional header which can be `nothing` or have any form accepted by the `FitsHeader`
constructor. If `hdr` is not specifed, the other keywords than `overwrite` are used to
build a header.

If `dest` is the name of a file which already exists, an error is thrown unless keyword
`overwrite = true` is specified. An alternative is to call `writefits!` which silently
overwrites existing files.

Note: `write(filename::AbstractString, ...)` is deprecated in favor of `writefits(...)`.
"""

writefits!(filename::AbstractString, data::WritableData; kwds...) =
    writefits(filename, data;  overwrite = true, kwds...)

writefits!(filename::AbstractString, hdr, data::WritableData) =
    writefits(filename, hdr, data; overwrite = true)

function writefits(filename::AbstractString,
                   data::WritableData; overwrite::Bool = false, kwds...)
    writefits(filename, FitsHeader(; kwds...), data; overwrite = overwrite)
end

function writefits(filename::AbstractString, hdr::Nothing,
                   data::WritableData; overwrite::Bool = false)
    writefits(filename, FitsHeader(), data; overwrite = overwrite)
end

function writefits(filename::AbstractString, hdr,
                   data::WritableData; overwrite::Bool = false)
    writefits(filename, FitsHeader(hdr), data; overwrite = overwrite)
end

function writefits(filename::AbstractString, hdr::FitsHeader,
                   data::WritableData; overwrite::Bool = false)
    (overwrite == false && ispath(filename)) && throw_file_already_exists(
        filename, "call `writefits!` or use `overwrite=true`")
    FitsFile(filename, (overwrite ? "w!" : "w")) do io
        write(io, filter(!is_structural, hdr), data)
    end
    nothing
end

write(io::FitsFile, data::WritableData; kwds...) =
    write(io, FitsHeader(; kwds...), data)

write(io::FitsFile, hdr::Nothing, data::WritableData) =
    write(io, FitsHeader(), data)

"""
    read(ReducedCalibration, src)
    readfits(ReducedCalibration, src)
    read(PreprocessingParameters, src)
    readfits(PreprocessingParameters, src)

read detector calibration parameters or preprocessing parameters
from source `src` (a file name or a FITS file instance).

"""
read(T::Type{<:WritableData}, filename::AbstractString) =
    readfits(T, filename)

readfits(T::Type{<:WritableData}, filename::AbstractString) =
    FitsFile(filename, "r") do io
        read(T, io)
    end

#------------------------------------------------------------------------------
#
# I/O methods for `ReducedCalibration`.
#

# Extend AstroFITS method to provide HDU name and revision number.
AstroFITS.hduname(::Type{<:ReducedCalibration}) =
    ("REDUCED-DETECTOR-CALIBRATION", 4)

function read(T::Type{<:ReducedCalibration}, io::FitsFile)
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(H -> matchvalue(H, "HDUNAME", name), io)
    k === nothing && error("no reduced detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    return read(T, hdu)
end

function read(::Type{ReducedCalibration}, hdu::FitsImageHDU{T}) where {T}
   return read(ReducedCalibration{float(T)}, hdu)
end

function read(::Type{ReducedCalibration{T}},
              hdu::FitsImageHDU{<:Any,N}) where {T<:AbstractFloat,N}
    return read(ReducedCalibration{T,N-1}, hdu)
end

function read(::Type{ReducedCalibration{T,N}},
              hdu::FitsImageHDU{<:Any,Np1}) where {T<:AbstractFloat,N,Np1}
    # Check HDUNAME, HDUVERS, and BITPIX.
    name, _ = hduname(ReducedCalibration)
    matchvalue(hdu, "HDUNAME", name) || error("bad HDUNAME, should be \"$name\"")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    1 ≤ version ≤ 4 || error("unsupported format revision $version")
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    Np1 == N + 1 || dimension_mismatch("incompatible number of dimensions")
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims = hdu.data_size
    @assert length(dims) == Np1
    if version ≤ 2
        n1 = 4 # number of fields before the sources
    elseif version == 3
        n1 = 5 # number of fields before the sources
    else
        n1 = 6
    end
    nsrc = dims[end] - n1
    nsrc ≥ 0 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = DetectorAxes{N}(hdr)
    src = [getvalue(String, hdr, "SRC$k", "") for k in 1:nsrc]

    # Read data and build instance.
    inds = colons(N)
    f = read(hdu, inds..., 1)
    z = read(hdu, inds..., 2)
    g = read(hdu, inds..., 3)
    σ = read(hdu, inds..., 4)
    σa = version > 3 ? read(hdu, inds..., 5) : FastUniformArray(zero(T), dims[1:end-1])

    vpm = n1 ≥ 5 ? read(Array{Bool,N}, hdu, inds..., 5 + (version > 3 ? 1 : 0))  :
        FastUniformArray(true, dims[1:end-1])
    s = [read(hdu, inds..., n1 + k) for k in 1:nsrc]
    return ReducedCalibration{T}(roi, f, z, g, σ, σa, s, src, vpm)
end

function write(io::FitsFile, hdr::FitsHeader,
               data::ReducedCalibration{T,N}) where {T,N}
    # Create HDU.
    dims = size(data)
    nsrc = length(data.s)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 6 + nsrc)

    # Write header.
    name, vers = hduname(data)
    hdu["HDUNAME"] = (name, "reduced detector calibration")
    hdu["HDUVERS"] = (vers, "version of this format")
    hdu["algo"] = ("$(data.algo)", "algorithm in this calibration")
    merge!(hdu, DetectorAxes(data))
    hdu["FRAME1"] = ("score", "co-log-likelihood (f)")
    hdu["FRAME2"] = ("bias", "[ADU] constant bias (z)")
    hdu["FRAME3"] = ("gain", "[electron/ADU] detector gain (g)")
    hdu["FRAME4"] = ("ron", "[ADU] readout-noise (sigma)")
    hdu["FRAME5"] = ("DITron", "[ADU/√s] DIT dependent readout-noise (sigma)")
    hdu["FRAME6"] = ("valid", "valid pixels map (vpm) (1=validpixel)")
    for k ∈ eachindex(data.src)
        hdu["FRAME$(6+k)"] = (data.src[k], "[ADU/s]")
    end
    for k ∈ 1:nsrc
        hdu["SRC$k"] = data.src[k]
    end
    merge!(hdu, hdr)

    # Write data array.
    tick = Ticker(1, prod(dims))
    write(hdu, data.f; first = tick())
    write(hdu, data.z; first = tick())
    write(hdu, data.g; first = tick())
    write(hdu, data.σ; first = tick())
    write(hdu, data.σa; first = tick())
    write(hdu, data.vpm; first = tick())
    for arr ∈ data.s
        write(hdu, arr; first = tick())
    end
    return io
end


#------------------------------------------------------------------------------
#
# I/O methods for `CalibrationData`.
#

# Extend AstroFITS method to provide HDU name and revision number.
AstroFITS.hduname(::Type{<:CalibrationData}) = ("DETECTOR-CALIBRATION-DATA", 1)

function write(io::FitsFile, hdr::FitsHeader, data::CalibrationData{T,N}) where {T,N}

    # we write `stat` first because it is an image HDU, so it can be primary
    # we merge `roi` in its header
    # we merge `hdr` also, we do this only in the primary HDU

    # first stat (remember, data.stat is a vector of statistics)
    name, vers = ("DETECTOR-CALIBRATION-DATA-STAT", 1)
    H = FitsHeader(
        "EXTNAME" => hduname(CalibrationData)[1],
        "HDUNAME" => hduname(CalibrationData)[1],
        "HDUVERS" => hduname(CalibrationData)[2])
    merge!(H, data.roi)
    merge!(H, hdr)
    filter!(!is_structural, H)
    group_id = string(1)
    write(io, H, data.stat[1], group_id)

    # other stats
    for i in 2:length(data.stat)
        group_id = string(i)
        write(io, FitsHeader(), data.stat[i], group_id)
    end

    # stat_index
    H = FitsHeader("EXTNAME" => "stat_index", "HDUNAME" => "stat_index")
    entries = collect(data.stat_index)
    write(io, H, [
        "CAT" => map(e -> e[1][1], entries),
        "TIME" => map(e -> e[1][2], entries),
        "INDEX" => map(e -> e[2], entries) ])

    # src_to_cat
    H = FitsHeader("EXTNAME" => "src_to_cat", "HDUNAME" => "src_to_cat")
    write(io, H, data.src_to_cat)

    # cat_index
    H = FitsHeader("EXTNAME" => "cat_index", "HDUNAME" => "cat_index")
    entries = collect(data.cat_index)
    write(io, H, [
        "CAT" => map(e -> e[1], entries),
        "INDEX" => map(e -> e[2], entries) ])

    # src_index
    H = FitsHeader("EXTNAME" => "src_index", "HDUNAME" => "src_index")
    entries = collect(data.src_index)
    write(io, H, [
        "SRC" => map(e -> e[1], entries),
        "INDEX" => map(e -> e[2], entries) ])

    return io
end

# type of float and number of axes are found in the header keywords
function read(::Type{CalibrationData}, io::FitsFile)

    # roi
    hdu_stat1_moment1 = OnlineSampleStatistics.find_stat_hdus(io, "1")[1][1]
    roi = DetectorAxes(hdu_stat1_moment1)

    T = hdu_stat1_moment1.data_eltype
    N = length(roi)

    # stat_index
    D = read(io["stat_index"])
    stat_index = Dict( (c,t) => i for (c,t,i) in zip(D["CAT"], D["TIME"], D["INDEX"]))

    # stat
    nb_stats = length(stat_index)
    stat = Vector{IndependentStatistic{T,N,2}}(undef, nb_stats)
    for i in 1:nb_stats
        group_id = string(i)
        stat[i] = read(IndependentStatistic, io, group_id)
    end

    # null
    null = zeros(T, size(roi))

    # src_to_cat
    src_to_cat = read(io["src_to_cat"])

    # cat_index
    D = read(io["cat_index"])
    cat_index = Dict( c => i for (c,i) in zip(D["CAT"], D["INDEX"]))

    # src_index
    D = read(io["src_index"])
    src_index = Dict( s => i for (s,i) in zip(D["SRC"], D["INDEX"]))

    return CalibrationData{T,N}(roi, stat_index, stat, null, src_to_cat, cat_index, src_index)
end



#------------------------------------------------------------------------------
#
# I/O methods for `SimpleCalibration`.
#

# Extend AstroFITS method to provide HDU name and revision number.
AstroFITS.hduname(::Type{<:SimpleCalibration}) =
    ("SIMPLE-DETECTOR-CALIBRATION", 1)

function read(T::Type{<:SimpleCalibration}, io::FitsFile)
    # Find HDU with calibration parameters.
    name, vers = hduname(T)
    k = findfirst(H -> matchvalue(H, "HDUNAME", name), io)
    k === nothing && error("no simple detector calibration found")
    hdu = io[k]
    isa(hdu, FitsImageHDU) || error("unexpected non-IMAGE HDU")
    return read(T, hdu)
end

function read(::Type{SimpleCalibration}, hdu::FitsImageHDU{T}) where {T}
   return read(SimpleCalibration{float(T)}, hdu)
end

function read(::Type{SimpleCalibration{T}},
              hdu::FitsImageHDU{<:Any,N}) where {T<:AbstractFloat,N}
    return read(SimpleCalibration{T,N-1}, hdu)
end

function read(::Type{SimpleCalibration{T,N}},
              hdu::FitsImageHDU{<:Any,Np1}) where {T<:AbstractFloat,N,Np1}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, _ = hduname(SimpleCalibration)
    matchvalue(hdu, "HDUNAME", name) || error("bad HDUNAME, should be \"$name\"")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    version == 1 || error("unsupported format revision $version")
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save simple calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    Np1 == N + 1 || dimension_mismatch("incompatible number of dimensions")
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims = hdu.data_size
    @assert length(dims) == Np1
    dims[end] == 5 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = DetectorAxes{N}(hdr)
    Δt = hdr["EXPTIME"].float

    # Read data and build instance.
    inds = colons(N)
    f = read(hdu, inds..., 1)
    a = read(hdu, inds..., 2)
    b = read(hdu, inds..., 3)
    g = read(hdu, inds..., 4)
    σ = read(hdu, inds..., 5)
    return SimpleCalibration{T,N}(roi, Δt, f, a, b, g, σ)
end

function write(io::FitsFile, hdr::FitsHeader,
               data::SimpleCalibration{T,N}) where {T,N}
    # Create HDU.
    dims = size(data)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 5)

    # Write header.
    name, vers = hduname(data)
    hdu["HDUNAME"] = (name, "simple detector calibration")
    hdu["HDUVERS"] = (vers, "version of this format")
    hdu["EXPTIME"] = (exposuretime(data), "[s] exposure time")
    merge!(hdu, DetectorAxes(data))
    merge!(hdu, hdr)

    # Write data.
    tick = Ticker(1, prod(dims))
    write(hdu, data.f; first = tick())
    write(hdu, data.a; first = tick())
    write(hdu, data.b; first = tick())
    write(hdu, data.g; first = tick())
    write(hdu, data.σ; first = tick())
    return io
end

#------------------------------------------------------------------------------
#
# I/O methods for `PreprocessingParameters`.
#

# Extend AstroFITS method to provide HDU name and revision number.
AstroFITS.hduname(::Type{<:PreprocessingParameters}) =
    ("DETECTOR-PREPROCESSING-PARAMETERS", 1)

function read(T::Type{<:PreprocessingParameters}, io::FitsFile)
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

function read(::Type{PreprocessingParameters}, hdu::FitsImageHDU{T}) where {T}
   return read(PreprocessingParameters{float(T)}, hdu)
end

function read(::Type{PreprocessingParameters{T}},
              hdu::FitsImageHDU{<:Any,N}) where {T<:AbstractFloat,N}
    return read(PreprocessingParameters{T,N-1}, hdu)
end

function read(::Type{PreprocessingParameters{T,N}},
              hdu::FitsImageHDU{<:Any,Np1}) where {T<:AbstractFloat,N,Np1}
    # Check HDUNAME, HDUVERS and BITPIX.
    name, _ = hduname(T)
    matchvalue(hdu, "HDUNAME", name) || error("bad HDUNAME, should be \"$name\"")
    version = getvalue(Int, hdu, "HDUVERS", 0)
    version == 1 || error(string("unsupported format revision ", version))
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    Np1 == N + 1 || dimension_mismatch("incompatible number of dimensions")
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims = hdu.data_size
    @assert length(dims) == Np1
    dims[end] == 4 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = DetectorAxes{N}(hdr)
    Δt = hdr["EXPTIME"].float

    # Read data and build instance.
    inds = colons(N)
    a = read(hdu, inds..., 1)
    b = read(hdu, inds..., 2)
    q = read(hdu, inds..., 3)
    r = read(hdu, inds..., 4)
    return PreprocessingParameters{T}(roi, Δt, a, b, q, r)
end

function write(io::FitsFile, hdr::FitsHeader,
               data::PreprocessingParameters{T,N}) where {T,N}
    # Create HDU.
    dims = size(data)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 4)

    # Write header.
    name, vers = hduname(data)
    hdu["HDUNAME"] = (name, "pre-processing parameters")
    hdu["HDUVERS"] = (vers, "version of this format")
    hdu["EXPTIME"] = (exposuretime(data), "[s] exposure time")
    merge!(hdu, DetectorAxes(data))
    merge!(hdu, hdr)

    # Write data.
    tick = Ticker(1, prod(dims))
    write(hdu, data.a; first = tick())
    write(hdu, data.b; first = tick())
    write(hdu, data.q; first = tick())
    write(hdu, data.r; first = tick())
    return io
end

#------------------------------------------------------------------------------
#
# I/O methods for `SampleStatistics`.
#

AstroFITS.hduname(::Type{<:SampleStatistics}) =
    ("DETECTOR-SAMPLE-STATISTICS", 1)

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

function _read1(T::Type{<:SampleStatistics}, hdu::FitsHDU)
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
        roi = ntuple(i -> DetectorAxis(dims[i]; off = off[i], bin = bin[i]), N) # TODO wrap in DetectorAxes
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
               DetectorAxis(dims[2]; off = yoff, bin = 1)) # TODO wrap in DetectorAxes
        return samples, exptime, roi
    end
    if exposure isa AbstractFloat && xbin isa Int && ybin isa Int
        # Assume exposure time in seconds.
        mesg == "" || error(mesg)
        exptime = Float64(exposure)
        roi = (DetectorAxis(dims[1]; off = xoff, bin = xbin),
               DetectorAxis(dims[2]; off = yoff, bin = ybin)) # TODO wrap in DetectorAxes
        return samples, exptime, roi
    end
    return nothing
end

function _read2(T::Type{<:SampleStatistics}, hdu::FitsImageHDU,
                samples::Integer, Δt::Float64,
                roi::NTuple{N,DetectorAxis}) where {N} # TODO NTuple{N,DetectorAxis}->DetectorAxes{N}
    # Read data and build instance.
    inds = colons(N)
    avg = read(hdu, inds..., 1)
    std = read(hdu, inds..., 2)
    return T(avg, std, samples, Δt, roi)
end

function write(io::FitsFile, hdr::FitsHeader,
               data::SampleStatistics{T,N}) where {T,N}
    # Create HDU.
    dims = size(data)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 2)

    # Write header.
    name, vers = hduname(data)
    hdu["HDUNAME"] = (name, "detector sample statistics")
    hdu["HDUVERS"] = (vers, "version of this format")
    hdu["EXPTIME"] = (exposuretime(data), "[s] exposure time")
    hdu["SAMPLES"] = (nobs(data), "number of samples")
    merge!(hdu, DetectorAxes(data))
    merge!(hdu, hdr)

    # Write data.
    tick = Ticker(1, prod(dims))
    write(hdu, mean(data); first = tick())
    write(hdu, std( data); first = tick())
    return io
end
