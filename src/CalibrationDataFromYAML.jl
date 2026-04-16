function CalibrationData{T}(low_yml::Union{AbstractDict,AbstractString};
                            basedir::AbstractString=pwd()) where {T<:AbstractFloat}

    if low_yml isa AbstractString
        low_yml = YAML.load_file(low_yml)
    end
   
    roi = Tuple(map(low_yml["roi"]) do axis
        DetectorAxis(
            axis["length"]
            ; off=get(axis,"offset",0), bin=get(axis,"bin",1), step=get(axis,"step",1))
    end)
    
    calib_cats = CalibrationCategory[
        CalibrationCategory(catname, Meta.parse(cat["sources"]))
        for (catname, cat) in low_yml["categories"] ]
    
    calibrationdata = CalibrationData{T}(roi, calib_cats)
    
    for (catname, cat) in low_yml["categories"]
        for (Δt, fitspaths) in cat["files"]
            for fitspath in fitspaths
                if !isabspath(fitspath)
                    fitspath = normpath(basedir, fitspath)
                end
                if !isfile(fitspath)
                    argument_error("Incorrect filepath \"$fitspath\".")
                end
                ext = get(cat, "datahdu", get(low_yml, "datahdu", 1))
                fitsdata = read_hdu_calibdata(fitspath, T, ext, catname, Δt, roi)
                push!(calibrationdata, fitsdata)
            end
        end
    end
    
    calibrationdata
end

using OnlineSampleStatistics: build_from_rawmoments, get_rawmoments

function read_hdu_calibdata(fitspath::String,
                            typefloat::Type{T},
                            ext::Union{Integer,String},
                            catname::String,
                            Δt::Real,
                            roi::DetectorAxes{N}
) where {T,N}
    foreach(roi) do axis
        (axis.off == 0 && axis.bin == 1 && axis.stp == 1) || error(
            "only trivial DetectorAxis are handled for now")
    end
    FitsFile(fitspath) do fits
        hdu = fits[ext]

        (hdu isa FitsImageHDU) || argument_error("HDU must be an image.")
        
        if OnlineSampleStatistics.isa_stat_hdu(hdu)
            stat = read(IndependentStatistic, fits; ext)
            # converting to correct type and order if needed
            if order(stat) > 2
                stat = build_from_rawmoments(
                    nobs(stat), (get_rawmoments(stat,1), get_rawmoments(stat,2)))
            end
            if !(stat isa IndependentStatistic{T})
                stat = build_from_rawmoments(
                    nobs(stat), (map(T, get_rawmoments(stat,1)), map(T, get_rawmoments(stat,2))))
            end
            return CalibrationDataStat{T,N}(catname, Δt, stat, roi)
        else
            data = read(Array{T}, hdu)
            if ndims(data) == N
                return CalibrationDataFrame{T,N}(catname, Δt, data; roi)
            elseif ndims(data) == N+1
                if size(data,N+1) == 1
                    return CalibrationDataFrame{T,N}(catname, Δt, reshape(data, Val(N)); roi)
                else
                    return CalibrationFrameSampler(data, catname, Δt; roi)
                end
            else
                dimension_mismatch(string(
                    "in filepath \"$filepath\", HDU \"$ext\" has $(ndims(data)) dimensions, ",
                    "whereas we expect $N or $(N+1) dimensions."))
            end
        end
    end
end

