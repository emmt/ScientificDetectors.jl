using YAML

function CalibrationData{T}(yaml_filepath::AbstractString;
                            basedir::AbstractString=pwd(),
                            reader::Function=reader_fits,
                            verb::Bool=false) where {T<:AbstractFloat}

    yml = YAML.load_file(yaml_filepath)
   
    roi = DetectorAxes(
        DetectorAxis(ax["length"]; off=ax["offset"], bin=ax["bin"], step=ax["step"])
        for ax in yml["roi"])
    
    cats = [
        CalibrationCategory(catname, Meta.parse(cat["sources"]))
        for (catname, cat) in yml["categories"] ]
    
    calibration_data = CalibrationData{T}(roi, cats; verb)
    
    datahdus = Dict(
        catname => haskey(cat, "datahdu") ? cat["datahdu"] : 1
        for (catname, cat) in yml["categories"])
    
    for fitspath in toto
        fitsdata = reader_fits(T, fitspath, roi, datahdus[titi])
        push!(calibration_data, fitsdata)
    end
    
    calibration_data
end

function reader_fits(::Type{T},
                     fitspath::AbstractString,
                     roi::DetectorAxes{N},
                     datahdu::Union{Integer,String}) where {T<:AbstractFloat,N}
    
    fitsdata = FitsFile(fitspath) do fitsfile
        
        hdu = fitsfile[datahdu]
        
        if hdu isa FitsImageHDU
            if hdu["HDUNAME"] == hduname(SampleStatistics)
                reader_fits_SampleStatistics(T, fitspath, datahdu, hdu, roi)
            else
                reader_fits_FitsImageHDU(T, fitspath, datahdu, hdu, roi)
            end
        end
    end
    
    fitsdata
end

function reader_fits_FitsImageHDU(::Type{T},
                                  fitspath::AbstractString,
                                  datahduindex::Union{Integer,String},
                                  hdu::FitsImageHDU{T},
                                  roi::DetectorAxes{N}) where {T<:AbstractFloat,N}

    #TODO: merge EasyFITS branch which implements binning
    foreach(roi) do roi_ax ; roi_ax.bin == 1 || error("not implemented yet") end

    hdu.data_ndims in (N, N+1) || dimension_mismatch(string(
    "For file \"$fitspath\" HDU \"$datahduindex\", ",
    "HDU has $(hdu.data_ndims) dimensions whereas provided ROI has $N dimensions. ",
    "ROI must address every HDU's axes, except the last one, which is the frames axis. ",
    "However it is tolerated that an HDU with only 1 frame has no frame axis."))

    # converting DetectorAxis to SubArrayIndex
    subarrayindices = ntuple(N) do d
        (1 + roi[d].off) : (roi[d].stp) : (roi[d].off + roi[d].len)
    end

    # adding frame dimension if needed
    subarrayindices = (N == hdu.data_ndims) ? subarrayindices : (subarrayindices..., Colon())
end



function reader_fits_SampleStatistics(::Type{T},
                                      fitspath::AbstractString,
                                      datahduindex::Union{Integer,String},
                                      hdu::FitsImageHDU{T},
                                      roi::DetectorAxes{N}) where {T<:AbstractFloat,N}

    hdu.data_ndims == N+1 || dimension_mismatch(string(
        "For file \"$fitspath\" HDU \"$datahduindex\", ",
        "HDU has $(hdu.data_ndims) dimensions whereas provided ROI has $N dimensions. ",
        "ROI must address every HDU's axes, except the last one, which is the sample axis."))

    samplestats = read(SampleStatistics{T}, hdu)
    
    roi == samplestats.roi || dimension_mismatch(string(
        "For file \"$fitspath\" hdu \"$datahduindex\", ",
        "SampleStatistics have existing ROI different from asked ROI: ",
        "SampleStatistics' ROI is `$(samplestats.roi)` and asked ROI is `$ROI`."))
        
    
end


