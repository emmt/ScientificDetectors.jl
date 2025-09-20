import Base: read, ==, write
import ScientificDetectors: write!, hduname, Ticker, OnlineStatistics


#=
==(x::CalibrationData, y::CalibrationData) =
    x.roi == y.roi &&
    x.stat_index == y.stat_index &&
    x.src_to_cat == y.src_to_cat &&
    x.cat_index == y.cat_index &&
    x.src_index == y.src_index &&
    all(i -> x.stat[i] == y.stat[i], eachindex(x.stat))=#

==(x::CalibrationData, y::CalibrationData) =
    all(fieldnames(CalibrationData)) do symb; getfield(x, symb) == getfield(y, symb) end


==(x::OnlineStatistics, y::OnlineStatistics) =
    all(fieldnames(OnlineStatistics)) do symb; getfield(x, symb) == getfield(y, symb) end

#------------------------------------------------------------------------------
#
# I/O methods for `CalibrationData`.
#

# Extend EasyFITS method to provide HDU name and revision number.
EasyFITS.hduname(::CalibrationData) = ("DETECTOR-CALIBRATION-DATA-STATISTICS", 1)

EasyFITS.write!(filename, d::CalibrationData) = 
    FitsFile(filename, "w!") do fits
        write(fits, FitsHeader(), d)
        nothing
    end

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

read(::Type{CalibrationData}, filename) = FitsFile(filename) do fits; read(CalibrationData, fits) end

# type of float and number of axes are found in the header keywords
function read(::Type{ScientificDetectors.CalibrationData}, io::FitsFile)

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

