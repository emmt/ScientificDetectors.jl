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
                FitsFile(fitspath) do fitsfile
                    ext = datahduindex[catname]
                    hdu = fitsfile[ext]
                    hdudata = try read_hdu_calibdata(T, hdu, catname, Δt, roi)
                              catch e
                                    error("problem with FITS file \"$fitspath\".")
                              end
                    push!(calibrationdata, hdudata)
                end
            end
        end
    end
    
    calibrationdata
end

function read_hdu_calibdata(typefloat::Type{T},
                            hdu::FitsHDU,
                            catname::String,
                            Δt::Real,
                            roi::DetectorAxes{N}
)::Union{CalibrationDataFrame, CalibrationFrameSampler, CalibrationDataStat} where {T,N}

    (hdu isa FitsImageHDU) || argument_error("HDU must be an image.")

    foreach(roi) do axis
        (axis.off == 0 && axis.bin == 1 && axis.stp == 1) || error(
            "only trivial DetectorAxis are handled for now")
    end

    if MultivariateOnlineStatistics.isa_stat_hdu(hdu)
        L = hdu.data_size[end]
        stat = read(IndependentStatistics{L,T,N}, hdu)
        return CalibrationDataStat{T,N}(catname, Δt, stat, roi)
    else
        data = read(Array{T}, hdu)
        if ndims(data) == N
            return CalibrationDataFrame{T,N}(catname, Δt, data; roi=roi)
        elseif ndims(data) == N+1
            if size(data,N+1) == 1
                return CalibrationDataFrame{T,N}(catname, Δt, reshape(data, Val(N)); roi=roi)
            else
                return CalibrationFrameSampler(data, catname, Δt; roi=roi)
            end
        else
            dimension_mismatch(
                "HDU should have $N or $(N+1) dimensions instead of $(ndims(data))")
        end
    end
end

