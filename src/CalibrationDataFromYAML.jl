function CalibrationData{T}(yml::Union{AbstractDict,AbstractString};
                            basedir::AbstractString=pwd(),
                            reader_fits::Function=reader_fits,
                            verb::Bool=false) where {T<:AbstractFloat}

    if yml isa AbstractString
        yml = YAML.load_file(yml)
    end
   
    roi = DetectorAxes(
        DetectorAxis(ax["length"]; off=ax["offset"], bin=ax["bin"], step=ax["step"])
        for ax in yml["roi"])
    
    cats = CalibrationCategory[
        CalibrationCategory(catname, Meta.parse(cat["sources"]))
        for (catname, cat) in yml["categories"] ]
    
    datahduindex = Dict(
        catname => haskey(cat, "datahdu") ? cat["datahdu"] : 1
        for (catname, cat) in yml["categories"])
    
    calibration_data = CalibrationData{T}(roi, cats; verb)
    
    for (catname, cat) in yml["categories"]
        for (Δt, fitspaths) in cat["files"]
            for fitspath in fitspaths
                fitspath = normpath(fitspath)
                fitspath = isabspath(fitspath) ? fitspath : joinpath(basedir, fitspath)
                iteratordata = reader_fits(T, fitspath, catname, Δt, roi, datahduindex[catname])
                push!(calibration_data, iteratordata)
                verb && @info "$catname $(Δt)s \"$fitspath\""
            end
        end
    end
    
    calibration_data
end


function reader_fits(::Type{T},
                     fitspath::AbstractString,
                     categoryname::AbstractString,
                     Δt::Real,
                     roi::DetectorAxes{N},
                     datahduindex::Union{Integer,String}) where {T<:AbstractFloat,N}
    
    fitspath = normpath(fitspath)
    
    isfile(fitspath) || argument_error("does not seems to be a file: \"$fitspath\".")
    
    fitsdata = FitsFile(fitspath) do fitsfile
        
        hdu = fitsfile[datahduindex]
        
        if hdu isa FitsImageHDU
            if hdu.hduname == EasyFITS.hduname(ImageStat)[1]
                reader_fits_ImageStat(T, hdu, categoryname, Δt, roi)
            else 
                isnothing(hdu.hduname) || isempty(hdu.hduname) || @warn string(
                    "Reading HDU as basic image, but it has HDU name $(hdu.hduname), in file ",
                    fitspath)
                reader_fits_FitsImageHDU(T, hdu, categoryname, Δt, roi)
            end
        else
            argument_error(string(
                "For file \"$fitspath\" HDU \"$datahduindex\", ",
                "unmanaged HDU type: `$(typeof(hdu))`"))
        end
    end
    
    fitsdata
end


function reader_fits_FitsImageHDU(::Type{T},
                                  hdu::FitsImageHDU,
                                  categoryname::AbstractString,
                                  Δt::Real,
                                  roi::DetectorAxes{N}) where {T<:AbstractFloat,N}

    hdu.data_ndims in (N, N+1) || dimension_mismatch(string(
    "For file \"$fitspath\" HDU \"$datahduindex\", ",
    "HDU has $(hdu.data_ndims) dimensions whereas provided ROI has $N dimensions. ",
    "ROI must address every axis of the HDU, except the last one, which is the frames axis. ",
    "However it is tolerated that an HDU with only 1 frame has no frame axis."))
        
    hduroi = get(DetectorAxes{N}, hdu)

    data =
        if hduroi == roi
            read(Array{T}, hdu)

        elseif roi ⊆ hduroi
            seqroi = hduroi[roi]
            
            #TODO: implement binning > 1
            all(axis -> axis.bin == 1, seqroi) || error("binning > 1 not implemented yet")
            
            # read only necessary data
            # converting DetectorAxes to SubArrayIndices
            indices = ntuple(d -> range([],seqroi[d],d), N)
            # adding frame dimension if present in the HDU (it is nearly always the case)
            if hdu.data_ndims == N+1
                indices = (indices..., Colon())
            end
            read(Array{T}, hdu, indices)
            
        else
            dimension_mismatch(string(
                "File \"$(hdu.file.path)\" HDU \"$(hdu.number)\" has dimensions `$hduroi` ",
                "which is incompatible with asked ROI `$roi`."))
        end
    
    iterator =
        if ndims(data) == N
            CalibrationDataFrame{T,N}(categoryname, Δt, data; roi=roi)
            
        elseif ndims(data) == N+1
            if size(data,N+1) == 1
                data = reshape(data, size(data)[1:N])
                CalibrationDataFrame{T,N}(categoryname, Δt, data; roi=roi)
            elseif size(data,N+1) > 1
                CalibrationFrameSampler(data, categoryname, Δt; roi=roi)
            else
                dimension_mismatch("frame axis is present but there is no frames")
            end
            
        else
            error("should never happen")
        end
        
    iterator
end


function reader_fits_ImageStat(::Type{T},
                               hdu::FitsImageHDU,
                               categoryname::AbstractString,
                               Δt::Real,
                               roi::DetectorAxes{N}) where {T<:AbstractFloat,N}

    hdu.data_ndims == N+1 || dimension_mismatch(string(
        "For file \"$(hdu.file.path)\" HDU \"$(hdu.number)\", ",
        "HDU has $(hdu.data_ndims) dimensions whereas provided ROI has $N dimensions. ",
        "ROI must address every axis of the HDU, except the last one, which is the ",
        "statistics axis."))

    imagestat = read(ImageStat{T,N}, hdu)
    
    Δt ≈ exposuretime(imagestat) || @error string(
        "Exposure times are different between the one given in the YAML `$Δt` ",
        "and the one found in SampleStatistics HDU of the file `$(exposuretime(imagestat))`.")
    
    # if needed, cut `imagestat` to `roi`
    stat =
        if roi == DetectorAxes(imagestat)
            # data has already the asked ROI
            imagestat.stat
            
        elseif roi ⊆ DetectorAxes(imagestat)
            seqroi = DetectorAxes(imagestat)[roi]
            
            #TODO: implement binning > 1
            all(axis -> axis.bin == 1, seqroi) || error("binning > 1 not implemented yet")
            
            moment1 = imagestat.stat.s[1][seqroi]
            moment2 = imagestat.stat.s[2][seqroi]
            OnlineStatistics{T,N}((moment1, moment2), nobs(imagestat))
        else
            dimension_mismatch(string(
                "File \"$(hdu.file.path)\" HDU \"$(hdu.number)\" has dimensions ",
                "`$(DetectorAxes(imagestat))` which is incompatible with asked ROI `$roi`."))
        end
    
    CalibrationDataStat{T,N}(categoryname, Δt, stat, roi)
end


