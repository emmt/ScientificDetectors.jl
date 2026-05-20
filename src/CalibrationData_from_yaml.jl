function CalibrationData{T}(yml::AbstractDict;
                            basedir::AbstractString=pwd()) where {T<:AbstractFloat}

    # check YAML input, give hints to the user in case of error
    haskey(yml, "roi") || argument_error("`yml` must have field \"roi\"")
    yml["roi"] isa AbstractVector || argument_error("field \"roi\" must be a sequence")
    isempty(yml["roi"]) && argument_error("field \"roi\" must have at least one element")
    for (i,axis) in enumerate(yml["roi"])
        axis isa AbstractDict || argument_error("axis n°$i of roi must be a dictionnary")
        haskey(axis, "length") || argument_error("axis n°$i of roi must have the field \"length\"")
        for field in keys(axis)
            field in ("length", "offset", "bin", "step") || argument_error(string(
                "axis n°$i of roi has unknown field \"$field\", accepted fields are: ",
                "\"length\", \"offset\", \"bin\", and \"step\""))
        end
    end
    haskey(yml, "categories") || argument_error("`yml` must have field \"categories\"")
    yml["categories"] isa AbstractDict || argument_error(
        "field \"categories\" must be a dictionnary")
    isempty(yml["categories"]) && argument_error("`yml` must have at least one category")
    for (name,category) in yml["categories"]
        category isa AbstractDict || argument_error("category \"$name\" must be a dictionnary")
        haskey(category, "sources") || argument_error(
            "category \"$name\" must have the field \"sources\"")
        haskey(category, "files") || argument_error(
            "category \"$name\" must have the field \"files\"")
        category["files"] isa AbstractDict || argument_error(
            "field \"files\" of category \"$name\" must be a dictionnary")
        for (Δt,subfiles) in category["files"]
            Δt isa Real || argument_error(
                "in category \"$name\", exptime \"$(Δt)\" must be a number")
            subfiles isa AbstractVector || argument_error(
                "in category \"$name\", in exptime \"$(Δt)\", list of files must be a sequence")
            for (i,path) in enumerate(subfiles)
                path isa AbstractString || argument_error(
                    "in category \"$name\", in exptime \"$(Δt)\", element n°$i must be a path")
                # also check if file is readable, better fail early
                path = normpath(basedir, path)
                isfile(path) || argument_error(
                    "in category \"$name\", in exptime \"$(Δt)\", path \"$path\" must be a file")
                isreadable(path) || argument_error(
                    "in category \"$name\", in exptime \"$(Δt)\", file \"$path\" must be readable")
            end
        end
        for field in keys(category)
            field in ("sources", "files", "datahdu") || argument_error(string(
                "in category \"$name\", unknown field \"$field\", accepted fields are: ",
                "\"sources\", \"files\", and \"datahdu\"."))
        end
    end

    nb_files = sum(yml["categories"]) do (name,category); length(category["files"]) end
    isempty(nb_files) && argument_error("`yml` must have at least one calibration file")

    roi = DetectorAxes(map(yml["roi"]) do axis
        DetectorAxis(axis["length"]
                     ; off=get(axis,"offset",0),
                       bin=get(axis,"bin",1),
                       step=get(axis,"step",1))
    end...)
    
    categories = CalibrationCategory[
        CalibrationCategory(catname, Meta.parse(cat["sources"]))
        for (catname, cat) in yml["categories"] ]
    
    calib_data = CalibrationData{T}(roi, categories)
    
    progress = Progress(nb_files; desc="reading calibration files")
    for (catname, category) in yml["categories"]
        for (Δt, fitspaths) in category["files"]
            for fitspath in fitspaths
                fitspath = normpath(basedir, fitspath)
                ext = get(category, "datahdu", get(yml, "datahdu", 1))
                file_data = read_file_data(fitspath, T, ext, catname, Δt, roi)
                push!(calib_data, file_data)
            end
            next!(progress)
        end
    end
    finish!(progress)
    
    calib_data
end

CalibrationData{T}(yml_path::AbstractString;
                   basedir::AbstractString=pwd()) where {T<:AbstractFloat} =
    CalibrationData{T}(YAML.load_file(yml_path); basedir)

function read_file_data(fitspath::String,
                        typefloat::Type{T},
                        ext::Union{Integer,String},
                        catname::String,
                        Δt::Real,
                        roi::DetectorAxes{N}
) where {T,N}
    foreach(roi) do axis
        axis.bin == 1 || error("for DetectorAxis, only bin=1 is handled for now")
    end
    FitsFile(fitspath) do fits
        hdu = fits[ext]

        (hdu isa FitsImageHDU) || argument_error(
            "in FITS file \"$fitspath", HDU \"$ext\" must be an image")

        if hdu.data_ndims == N
            frame = read(Array{T,N}, hdu, axes(roi)...)
            return CalibrationDataFrame{T,N}(catname, Δt, frame; roi)

        elseif hdu.data_ndims == N+1

            if hdu.data_size[N+1] == 1
                frame = read(Array{T,N}, hdu, axes(roi)..., 1)
                return CalibrationDataFrame{T,N}(catname, Δt, frame; roi)

            else
                cube = read(Array{T,N+1}, hdu, axes(roi)..., :)
                return CalibrationFrameSampler(cube, catname, Δt; roi)
            end

        else
            dimension_mismatch(string(
                "in FITS file \"$fitspath\", HDU \"$ext\" has $(hdu.data_ndims) dimensions, ",
                "whereas we expect $N or $(N+1) dimensions for roi $roi"))
        end
    end
end