using EasyFITS: FitsFile, FitsImageHDU
import Base: read

function CalibrationData{T}(yml::Union{AbstractDict,AbstractString};
                            basedir::AbstractString=pwd(),
                            read_fits_samples::Function=default_read_fits_samples,
                            verb::Bool=false) where {T<:AbstractFloat}

    # if `yml` is a `String`, it is a path to a YAML file, and we parse it
    if yml isa AbstractString
        yml = YAML.load_file(yml)
    end

    # checking some YAML fields
    haskey(yml, "categories") || argument_error("YAML file must have a `categories` entry.")
    for (name, cat) in yml["categories"]
        haskey(cat, "sources") || argument_error("YAML category $name must have a `sources` entry.")
        haskey(cat, "files") || argument_error("YAML category $name must have a `files` entry.")
    end

    roi = resolve_roi(yml; basedir)

    cats = [ CalibrationCategory(name, Meta.parse(cat["sources"]))
             for (name, cat) in yml["categories"]                   ]

    calibdata = CalibrationData{T}(roi, cats; verb)

    for (catname, cat) in yml["categories"]
        for (Δt, filelist) in cat["files"]
            for filepath in filelist
                resolved_filepath = joinpath(basedir, filepath)
                datahdu = haskey(cat, "datahdu") ? Int(cat["datahdu"]) : 1
                samples = read_fits_samples(T, resolved_filepath, catname, Δt, roi, datahdu)
                push!(calibdata, samples)
                verb && @info "$catname $(Δt)s \"$resolved_filepath\""
            end
        end
    end

    calibdata
end

function default_read_fits_samples(
    ::Type{T}, resolved_filepath::String, catname::String, Δt::Real, roi::DetectorAxes{N}, datahdu
) where {T<:AbstractFloat, N}

    isreadable(resolved_filepath) || argument_error("cannot read file: \"$resolved_filepath\".")

    FitsFile(resolved_filepath) do fits
        hdu = fits[datahdu]

        data = read(hdu, roi)

        nbframes = size(data)[end]

        # return an iterator of Calibration Data Frames
        if nbframes == 1
            # drop the frames dimension
            reshaped_data = reshape(data, size(data[1:end-1]))
            CalibrationDataFrame{T,N}(catname, Δt, reshaped_data; roi)
        else
            # CalibrationFrameSampler works only for nbframes > 1
            CalibrationFrameSampler(data, catname, Δt; roi)
        end
    end
end

function read(hdu::FitsImageHDU{T,M}, roi::DetectorAxes{N}) where {T,M,N}
    (M == N+1) || argument_error(
        "number of dimension in `roi` ($N) must be one less than in `hdu` ($M).")
    #TODO: implement binning
    any(ax -> ax.bin != 1, roi) && error("bin != 1 not implemented")
    indices = axes(roi) # only work with bin = 1
    indices_with_frames = (indices..., :) # we add the frames dimension
    read(hdu, indices_with_frames)
end

function resolve_roi(yml::AbstractDict; basedir::AbstractString)

    # default value for `roi` is "FULL" (the whole detector surface)
    if !haskey(yml, "roi")
        yml["roi"] = "FULL"
    end

    # if `roi` is "FULL", find the first file and use its size as ROI
    if yml["roi"] == "FULL"
        first_file_size = get_first_file_size(yml; basedir)
        # !! we do not use the last axis, as it is the frames axis !!
        axes = first_file_size[1:end-1]
        return DetectorAxes(axes...)

    # if `roi` is a `String` but not "FULL", it is a range we try to parse
    elseif yml["roi"] isa AbstractString
        #TODO: parse ranges, a regex would do that easily
        error("not implemented yet")

    # if `roi` is a YAML AbstractVector, then it must be a vector of axes,
    # an axis being the YAML description of a DetectorAxis, expressed as a Dict
    elseif yml["roi"] isa AbstractVector
       axes = map(yml["roi"]) do axis
            axis isa AbstractDict  || argument_error("YAML `roi` axis should be a dict too")
            off  = haskey(axis, "off")  ? Int(axis["off"])  : 0
            step = haskey(axis, "step") ? Int(axis["step"]) : 1
            bin  = haskey(axis, "bin")  ? Int(axis["bin"])  : 1
            length = haskey(axis, "length") ? Int(axis["length"]) : argument_error(
                "YAML `roi` axis must have a `length` field")
            DetectorAxis(length; off, bin, step)
       end
       return DetectorAxes(axes...)

    else
        error("cannot resolve `roi`.")
    end
end

function get_first_file_size(yml::AbstractDict; basedir::AbstractString)
    size_found = nothing
    for (name, cat) in yml["categories"]
        for (Δt, filelist) in cat["files"]
            for filepath in filelist
                FitsFile(joinpath(basedir, filepath)) do fits
                    datahdu = haskey(cat, "datahdu") ? Int(cat["datahdu"]) : 1
                    size_found = fits[datahdu].data_size
                end
                break
            end
           isnothing(size_found) || break
        end
       isnothing(size_found) || break
    end
    isnothing(size_found) && argument_error(
        "Cannot find first FITS file size, because there is zero FITS file listed")
    size_found
end

