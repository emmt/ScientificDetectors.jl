module SphereData

using ScientificDetectors
using ScientificDetectors: CalibrationData
using FITSIO, EasyFITS

const IndexRange = Union{Colon,AbstractRange{<:Integer}}

struct CalibrationInformation
    path::String  # FITS file
    nframes::Int  # number of frames
    Δt::Float64   # exposure time (seconds)
    cat::String   # category name
end

"""
    readcalibrations(T, lst, xrng = :, yrng = :)

reads calibration data described in list `lst`.  Argument `T` is the
floating point type to use for computations.  Arguments 'xrng' and 'yrng'
are to select a sub-region in the calibration frames.

Typical usage (`dir` is the directory where are located calibration data):

    using ScientificDetectors, SphereData
    lst = SphereData.listcalibrations(dir)
    dat = SphereData.readcalibrations(Float32, lst, (401:600, 401:600))
    cal = ScientificDetectors.ReducedCalibration(dat)
    write("calib.fits", cal)

"""
function readcalibrations(::Type{T},
                          list::AbstractVector{CalibrationInformation},
                          inds::NTuple{N,IndexRange}) where {T<:AbstractFloat,N}
    function producer(chn::Channel)
        cnt = 0
        roi = Ref{DetectorAxes{N}}()
        for calib in list
            hdu = FITS(calib.path)[1] :: ImageHDU
            naxis = ndims(hdu)
            nframes = calib.nframes
            if naxis == N && nframes == 1
                if cnt == 0
                    roi[] = generate_detector_axes(size(hdu), inds)
                end
                put!(chn,
                     CalibrationDataFrame{T,N}(calib.cat, calib.Δt,
                                               read(hdu, inds...);
                                               roi = roi[]))
                cnt += 1
            elseif naxis == N+1
                if cnt == 0
                    roi[] = generate_detector_axes(size(hdu)[1:N], inds)
                end
                for j in 1:calib.nframes
                    put!(chn,
                         CalibrationDataFrame{T,N}(calib.cat, calib.Δt,
                                                   read(hdu, inds..., j);
                                                   roi = roi[]))
                    cnt += 1
                end
            else
                error("other dimensions than $(N)D and $(N+1)D not implemented")
            end
        end
        close(chn)
    end
    CalibrationData{T,N}(Channel{CalibrationDataFrame{T,N}}(producer))
end

generate_detector_axis(dim::Int, ::Colon) = DetectorAxis(dim)
generate_detector_axis(dim::Int, rng::OrdinalRange{<:Integer,<:Integer}) = begin
    1 ≤ first(rng) ≤ last(rng) ≤ dim || error("invalid range")
    DetectorAxis(length(rng); bin=step(rng), off=first(rng)-1)
end
generate_detector_axes(dims::Dims{N}, inds::NTuple{N,IndexRange}) where {N} =
    ntuple(i -> generate_detector_axis(dims[i], inds[i]), Val(N))

"""
    listfitsfiles(dir = pwd(), suffixes=(".fits", ".fits.gz",".fits.Z")) -> lst

yields the list of FITS files in directory `dir`, that is all files whose name
ends with one of the sufixes in `suffixes`.

"""
function listfitsfiles(dir::AbstractString = pwd(),
                       suffixes=(".fits", ".fits.gz",".fits.Z"))
    list = String[]
    for name in readdir(dir)
        path = joinpath(dir, name)
        isfile(path) || continue
        for sfx in suffixes
            if endswith(name, sfx)
                push!(list, path)
                break
            end
        end
    end
    return list
end

"""
    listcalibrations(dir = pwd()) -> cal

or

    listcalibrations(lst) -> cal

yields a vector of information about calibration data found in FITS files
in directory `dir` or in list of files `lst`.

Keyword `exptime` can be set with the name of the FITS card which stores the
exposure time (in seconds).  By default `exptime="ESO DET SEQ1 REALDIT"`.

Keyword `identify` can be set with a function that yields the category of a
given data file or `nothing` if it does not belong to a given category.  This
function is called as `identify(path,hdr)` with the name and header of the FITS
file.  By default `identify` uses the FITS card `"ESO DPR TYPE"` to identify
the category ignoring files for which this category is `"OBJECT"`.

"""
listcalibrations(dir::AbstractString = pwd(); kwds...) =
    listcalibrations(listfitsfiles(dir); kwds...)

function listcalibrations(list::AbstractVector{<:AbstractString};
                          identify::Function = identify_v1,
                          exptime::AbstractString = "ESO DET SEQ1 REALDIT")
    # We assume that there are only one exposure time (DIT)
    # and one cat of calibration per file.
    # FIXME: Reduce cubes!
    calib = CalibrationInformation[]
    width, height = -1, -1
    first = true
    for path in list
        hdr = read(FitsHeader, path)
        cat = identify(path, hdr)
        if cat === nothing
            continue
        end
        Δt = hdr[exptime]
        naxis = hdr["NAXIS"]
        if !(2 ≤ naxis ≤ 3)
            error("other dimensions than 2D and 3D not implemented")
        end
        thiswidth = hdr["NAXIS1"]
        thisheight = hdr["NAXIS2"]
        if first
            width = thiswidth
            height = thisheight
            first = false
        elseif width != thiswidth || height != thisheight
            error("file ", repr(path), " has incompatible dimensions")
        end
        nframes = (naxis == 2 ? 1 : hdr["NAXIS3"])
        push!(calib, CalibrationInformation(path, nframes, Δt, cat))
    end
    return calib
end

function identify_v1(path::AbstractString, hdr::FitsHeader)
    dprtype = hdr["ESO DPR TYPE"]
    #dprcatg = hdr["ESO DPR CATG"]
    if startswith(dprtype, "OBJECT")
        return nothing
    end
    return dprtype
end


function identify_v2(path::AbstractString, hdr::FitsHeader)
    name = basename(path)
    if endswith(name, ".gz")
        name = name[1:end-3]
    end
    if endswith(name, ".fits")
        name = name[1:end-5]
    end
    i = findfirst("IRD_", name)
    if i === nothing
        return nothing
    end
    type = get(hdr, "ESO DPR TYPE", nothing)
    if type === nothing || startswith(type, "OBJECT")
        return nothing
    end
    return name[last(i)+1:end]
end


end # module
