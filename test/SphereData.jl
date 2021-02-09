module SphereData

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
    dat = SphereData.readcalibrations(Float32, lst, 401:600, 401:600)
    cal = ScientificDetectors.ReducedCalibration(dat)
    write("calib.fits", cal)

"""
function readcalibrations(::Type{T},
                          list::AbstractVector{CalibrationInformation},
                          xrng::IndexRange = Colon(),
                          yrng::IndexRange = Colon()) where {T<:AbstractFloat}
    # FIXME: only work for images or cubes of images (OK for SPHERE)
    N = 2

    # Unpack calibration data
    nframes = 0
    for calib in list
        nframes += calib.nframes
    end
    slice = Array{Array{<:Real,2}}(undef, nframes)
    cat = Array{String}(undef, nframes)
    Δt = Array{T}(undef, nframes)
    i = 0 # frame index
    for calib in list
        hdu = FITS(calib.path)[1]
        naxis = read_key(hdu,"NAXIS")[1]
        if naxis == N && calib.nframes == 1
            i += 1
            slice[i] = read(hdu, xrng, yrng)
            cat[i] = calib.cat
            Δt[i] = calib.Δt
        elseif naxis == N+1
            for k in 1:calib.nframes
                i += 1
                slice[i] = read(hdu, xrng, yrng, k)
                cat[i] = calib.cat
                Δt[i] = calib.Δt
            end
        else
            error("other dimensions than $(N)D and $(N+1)D not implemented")
        end
    end

    # Convert data to a common type.
    P = promote_type(map(eltype, slice)...)
    data = Array{Array{P,N}}(undef, nframes)
    for i in 1:nframes
        data[i] = convert(Array{P,N}, slice[i])
    end
    return CalibrationData{P,N,T}(data, cat, Δt)
end

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
