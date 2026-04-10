function CalibrationData{T}(yml::Union{AbstractDict,AbstractString};
                            basedir::AbstractString=pwd()) where {T<:AbstractFloat}

    if yml isa AbstractString
        yml = YAML.load_file(yml)
    end
   
    haskey(yml, "roi") || throw(ArgumentError("yaml dict miss key \"roi\""))
    
    roi = Tuple(map(yml["roi"]) do ax
        ax isa AbstractDict  || throw(ArgumentError("yaml dict roi ax should be a Dict"))
        haskey(ax, "length") || throw(ArgumentError("yaml dict roi axis miss key \"length\""))
        DetectorAxis(ax["length"]
            ; off=get(ax,"offset",0), bin=get(ax,"bin",1), step=get(ax,"step",1))
    end)
    
    cats = CalibrationCategory[
        CalibrationCategory(catname, Meta.parse(cat["sources"]))
        for (catname, cat) in yml["categories"] ]
    
    datahduindex = Dict(
        catname => get(cat, "datahdu", get(yml, "datahdu", 1))
        for (catname, cat) in yml["categories"])
    
    calibrationdata = CalibrationData{T}(roi, cats)
    
    for (catname, cat) in yml["categories"]
        for (Δt, fitspaths) in cat["files"]
            for fitspath in fitspaths
                if !isabspath(fitspath)
                    fitspath = normpath(basedir, fitspath)
                end
                if !isfile(fitspath)
                    argument_error("Incorrect filepath \"$fitspath\".")
                end
                ext = datahduindex[catname]
                push!(calibrationdata, read_hdu_calibdata(fitspath, T, ext, catname, Δt, roi))
            end
        end
    end
    
    calibrationdata
end

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
            if eltype(get_moments(stat,1)) != T || order(stat) > 2
                moments = OnlineSampleStatistics.get_rawmoments(stat)
                stat = OnlineSampleStatistics.build_from_rawmoments(
                    nobs(stat), Tuple(T.(m) for m in moments[1:2]))
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
                    return CalibrationFrameSampler(data, catname, Δt; roi=roi)
                end
            else
                dimension_mismatch(
                    "HDU should have $N or $(N+1) dimensions instead of $(ndims(data))")
            end
        end
    end
end

