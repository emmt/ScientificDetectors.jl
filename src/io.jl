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
                                SimpleCalibration{T,N},
                                ImageStat{T,N}}

# HDU name and revision number only depend on the type of our objects..
AstroFITS.hduname(data::WritableData) = hduname(typeof(data))

"""
    write(dest, [hdr,] data; kwds...)
    writefits(dest, [hdr,] data; kwds...)

Write reduced detector calibration or preprocessing parameters `data` in `dest` which may
be a FITS file instance or the name of a new FITS file to create. Argument `hdr` is an
optional header which can be `nothing` or have any form accepted by the `FitsHeader`
constructor. If `hdr` is not specifed, the other keywords than `overwrite` are used to
build a header.

If `dest` is the name of a file which already exists, an error is thrown unless keyword
`overwrite = true` is specified. An alternative is to call `writefits!` which silently
overwrite existing files.

"""
write(filename::AbstractString, hdr, data::WritableData; overwrite::Bool = false) =
    writefits(filename, hdr, data; overwrite = overwrite)

write(filename::AbstractString, data::WritableData; kwds...) =
    writefits(filename, data; kwds...)

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
        write(io, hdr, data)
    end
    nothing
end

write(io::FitsFile, data::WritableData; kwds...) =
    write(io, FitsHeader(; kwds...), data)

write(io::FitsFile, hdr::Nothing, data::WritableData) =
    write(io, FitsHeader(), data)

write(io::FitsFile, hdr, data::WritableData) =
    write(io, FitsHeader(hdr), data)

"""
    read(ReducedCalibration, src)
    readfits(ReducedCalibration, src)
    read(PreprocessingParameters, src)
    readfits(PreprocessingParameters, src)

read detector calibration parameters or preprocessing parameters
from source `src` (a file name or a FITS file instance).

"""
read(T::Type{<:WritableData}, filename::AbstractString; kwds...) =
    readfits(T, filename; kwds...)

readfits(T::Type{<:WritableData}, filename::AbstractString; kwds...) =
    FitsFile(filename, "r") do io
        read(T, io; kwds...)
    end

#------------------------------------------------------------------------------
#
# I/O methods for `ReducedCalibration`.
#

# Extend AstroFITS method to provide HDU name and revision number.
AstroFITS.hduname(::Type{<:ReducedCalibration}) =
    ("REDUCED-DETECTOR-CALIBRATION", 3)

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
    1 ≤ version ≤ 3 || error("unsupported format revision $version")
    bitpix = hdu["BITPIX"].integer
    bitpix == -32 || bitpix == -64 ||
        @warn("To avoid loss of precision, save reduced calibration data "*
              "in floating point format, i.e. use BITPIX = -32 or -64.")

    # Check dimensions.
    Np1 == N + 1 || dimension_mismatch("incompatible number of dimensions")
    N ≥ 1 || dimension_mismatch("invalid number of dimensions")
    dims = hdu.data_size
    @assert length(dims) == Np1
    n1 = (version < 3 ? 4 : 5) # number of fields before the sources
    nsrc = dims[end] - n1
    nsrc ≥ 0 || dimension_mismatch("invalid last dimension")

    # Read header and retrieve contents.
    hdr = FitsHeader(hdu)
    roi = get(NTuple{N, DetectorAxis}, hdr)
    src = [getvalue(String, hdr, "SRC$k", "") for k in 1:nsrc]

    # Read data and build instance.
    inds = colons(N)
    f = read(hdu, inds..., 1)
    z = read(hdu, inds..., 2)
    g = read(hdu, inds..., 3)
    σ = read(hdu, inds..., 4)
    vpm = n1 ≥ 5 ? read(Array{Bool,N}, hdu, inds..., 5) :
        FastUniformArray(true, dims[1:end-1])
    s = [read(hdu, inds..., n1 + k) for k in 1:nsrc]
    return ReducedCalibration{T}(roi, f, z, g, σ, s, src, vpm)
end

function write(io::FitsFile, hdr::FitsHeader,
               data::ReducedCalibration{T,N}) where {T,N}
    # Create HDU.
    dims = size(data)
    nsrc = length(data.s)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 5 + nsrc)

    # Write header.
    name, vers = hduname(data)
    hdu["HDUNAME"] = (name, "reduced detector calibration")
    hdu["HDUVERS"] = (vers, "version of this format")
    merge!(hdu, DetectorAxes(data))
    hdu["FRAME1"] = ("score", "co-log-likelihood (f)")
    hdu["FRAME2"] = ("bias", "[ADU] constant bias (z)")
    hdu["FRAME3"] = ("gain", "[electron/ADU] detector gain (g)")
    hdu["FRAME4"] = ("ron", "[ADU] readout-noise (sigma)")
    hdu["FRAME5"] = ("valid", "valid pixels map (vpm) (1=validpixel)")
    for k ∈ eachindex(data.src)
        hdu["FRAME$(5+k)"] = (data.src[k], "[ADU/s]")
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
hduname(::Type{<:CalibrationData}) = ("DETECTOR-CALIBRATION-DATA-STATISTICS", 1)

function write(io::FitsFile, hdr::FitsHeader, data::CalibrationData{T,N}) where {T,N}

    # Create Primary HDU which contains roi, and statistics values
    dims = size(data.roi)
    statshdu = FitsImageHDU{T,N+2}(io, dims..., 2, length(data.stat))
    # `2` is for means and variances statistics
    # `length(data.stat)` is for the number of (cat,Δt) entries

    # Write header.
    name, vers = hduname(data)
    statshdu["EXTNAME"]  = "mean and variance stats"
    statshdu["HDUNAME"]  = (name, "statistics of detector calibration data")
    statshdu["HDUVERS"]  = (vers, "version of this format")
    merge!(statshdu, DetectorAxes(data)) # write `data.roi` as keywords
    merge!(statshdu, hdr)

    # Write data array.
    # Note that numbers of samples are written in another hdu
    tick = Ticker(1, prod(dims))
    for stat in data.stat
        write(statshdu, stat.s[1]; first = tick()) # means
        write(statshdu, stat.s[2]; first = tick()) # variances
    end

    # table HDU which contains (cat,Δt) entries and number of samples for each
    statindex ::Vector{Pair{Tuple{String,T},Int}} = sort(collect(data.stat_index) ; by=p->p[2])
    catnames  ::Vector{String}                    = map(x -> x.first[1], statindex)
    realdits  ::Vector{T}                         = map(x -> x.first[2], statindex)
    nsamples  ::Vector{Int}                       = [ stat.n for stat in data.stat ]
    longestcatname ::Int = maximum(length.(catnames))
    keyshdu = FitsTableHDU(io,
        :catname => (String, longestcatname),
        :realdit => T,
        :nsamples => Int)
    name, vers = ("DETECTOR-CALIBRATION-DATA-STATS-INDEX", 1)
    keyshdu["EXTNAME"] = "stats index"
    keyshdu["HDUNAME"]  = (name, "stats index of detector calibration data")
    keyshdu["HDUVERS"]  = (vers, "version of this format")
    write(keyshdu, :catname  => catnames)
    write(keyshdu, :realdit  => realdits)
    write(keyshdu, :nsamples => nsamples)

    # image HDU which contains `data.src_to_cat`
    dims = (length(data.cat_index), length(data.src_index))
    srctocathdu = FitsImageHDU{T,2}(io, dims...)
    name, vers = ("DETECTOR-CALIBRATION-DATA-SRC-TO-CAT", 1)
    srctocathdu["EXTNAME"]  = "src to cat"
    srctocathdu["HDUNAME"]  = (name, "src to cat matrix of detector calibration data")
    srctocathdu["HDUVERS"]  = (vers, "version of this format")
    write(srctocathdu, data.src_to_cat)

    # table HDU which contains `data.cat_index`
    cat_index = map(p->p[1], sort(collect(data.cat_index) ; by=p->p[2]))
    catindexhdu = FitsTableHDU(io, :cat_index => (String, longestcatname))
    name, vers = ("DETECTOR-CALIBRATION-DATA-CAT-INDEX", 1)
    catindexhdu["EXTNAME"]  = "cat index"
    catindexhdu["HDUNAME"]  = (name, "cat index of detector calibration data")
    catindexhdu["HDUVERS"]  = (vers, "version of this format")
    write(catindexhdu, :cat_index => cat_index)

    # table HDU which contains `data.src_index`
    src_index = map(p->p[1], sort(collect(data.src_index) ; by=p->p[2]))
    longestsrcname ::Int = maximum(length.(src_index))
    srcindexhdu = FitsTableHDU(io, :src_index => (String, longestsrcname))
    name, vers = ("DETECTOR-CALIBRATION-DATA-SRC-INDEX", 1)
    srcindexhdu["EXTNAME"]  = "src index"
    srcindexhdu["HDUNAME"]  = (name, "src index of detector calibration data")
    srcindexhdu["HDUVERS"]  = (vers, "version of this format")
    write(srcindexhdu, :src_index => src_index)
end

# type of float and number of axes are found in the header keywords
function read(::Type{CalibrationData}, io::FitsFile)

    statshdu    = io["mean and variance stats"] ::FitsImageHDU
    keyshdu     = io["stats index"]             ::FitsTableHDU
    srctocathdu = io["src to cat"]              ::FitsImageHDU
    catindexhdu = io["cat index"]               ::FitsTableHDU
    srcindexhdu = io["src index"]               ::FitsTableHDU

    T = statshdu.data_eltype ::Type{<:AbstractFloat}
    N = statshdu.data_ndims - 2

    roi = get(DetectorAxes{N}, statshdu)

    statsarr = read(Array{T}, statshdu)
    catnames = read(Vector{String}, keyshdu, :catname)
    realdits = read(Vector{T},      keyshdu, :realdit)
    nsamples = read(Vector{Int},    keyshdu, :nsamples)

    nstats ::Int = length(catnames)

    stat_index = Dict{Tuple{String,T},Int}()
    stat = Vector{OnlineStatistics{T,N}}(undef, nstats)

    for i in 1:nstats
        means = statsarr[ ((:) for _ in 1:N)..., 1, i ]
        vars  = statsarr[ ((:) for _ in 1:N)..., 2, i ]
        stat[i] = OnlineStatistics{T,N}((means, vars), nsamples[i])
        stat_index[ (catnames[i], realdits[i]) ] = i
    end

    null = zeros(T,size(roi))

    src_to_cat = read(Array{T,2}, srctocathdu)

    catindexarr = read(Vector{String}, catindexhdu, :cat_index)
    cat_index = Dict{String,Int}( cat => index for (index,cat) in enumerate(catindexarr))

    srcindexarr = read(Vector{String}, srcindexhdu, :src_index)
    src_index = Dict{String,Int}( src => index for (index,src) in enumerate(srcindexarr))

    return CalibrationData{T,N}(roi, stat_index, stat, null, src_to_cat, cat_index, src_index)
end

#------------------------------------------------------------------------------
#
# I/O methods for `SimpleCalibration`.
#

# Extend AstroFITS method to provide HDU name and revision number.
hduname(::Type{<:SimpleCalibration}) =
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
    name, _ = hduname(ReducedCalibration)
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
hduname(::Type{<:PreprocessingParameters}) =
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
    roi = get(NTuple{N,DetectorAxis}, hdr)
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

hduname(::Type{<:SampleStatistics}) =
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

function _read2(T::Type{<:SampleStatistics}, hdu::FitsImageHDU,
                samples::Integer, Δt::Float64,
                roi::NTuple{N,DetectorAxis}) where {N}
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


#------------------------------------------------------------------------------
#
# I/O methods for `ImageStat`.
#
hduname(::Type{<:ImageStat}) = ("IMAGE-STAT", 1)

function read(::Type{ImageStat{T,N}}, hdu::FitsImageHDU) where {T<:AbstractFloat,N}
    hdu.data_ndims == N+1 || dimension_mismatch("HDU has wrong number of dimensions")
    hdu.data_size[N+1] == 2 || dimension_mismatch("Number of moments must be 2")
    hdu.hduname == hduname(ImageStat)[1] || @warn "Not declared as an ImageStat HDU"

    Δt = hdu["EXPTIME"].float
    n  = hdu["NSAMPLES"].integer

    moment1 = read(Array{T,N}, hdu, ntuple(d->Colon(),N)..., 1)
    moment2 = read(Array{T,N}, hdu, ntuple(d->Colon(),N)..., 2)
    stat = OnlineStatistics{T,N}((moment1, moment2), n)

    roi = get(DetectorAxes{N}, hdu)

    ImageStat{T,N}(Δt, stat, roi)
end

function read(::Type{ImageStat}, hdu::FitsImageHDU)
    T = hdu.data_eltype
    N = hdu.data_ndims - 1  # last dimension is for mean and variance data
    read(ImageStat{T,N}, hdu)
end

function write(io::FitsFile, hdr::FitsHeader, data::ImageStat{T,N}) where {T,N}
    # Create HDU.
    dims = size(data.stat)
    hdu = FitsImageHDU{T,N+1}(io, dims..., 2)

    # Write header.
    merge!(hdu, hdr) # write user header first, so our keywords overwrite his
    hdu["HDUNAME"]  = (hduname(data)[1], "image statistics")
    hdu["HDUVERS"]  = (hduname(data)[2], "version of this format")
    hdu["EXPTIME"]  = (exposuretime(data), "[s] exposure time")
    hdu["NSAMPLES"] = (nobs(data), "number of samples")
    merge!(hdu, DetectorAxes(data))

    # Write data.
    tick = Ticker(1, prod(dims))
    write(hdu, data.stat.s[1]; first = tick())
    write(hdu, data.stat.s[2]; first = tick())
    return io
end

