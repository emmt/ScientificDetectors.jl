struct CalibrationDataStat{T<:Real,N}
    cat::String                 # Category.
    Δt::Float64                 # Exposure time.
    stat::OnlineStatistics{T,N} # Statistics.
    roi::DetectorAxes{N}  # Detector axes settings.
    
    function CalibrationDataStat{T,N}(cat::Category,
                                       Δt::Real,
                                       stat::OnlineStatistics{T,N},
                                       roi::DetectorAxes{N}=DetectorAxes(size(stat))
                                       ) where {T<:Real,N}
        Δt ≥ 0 || argument_error("exposure time must be nonnegative")
        size(stat) == size(roi) || dimension_mismatch(
            "statistics and region of interest have different sizes")
        obj = new{T,N}(cat, Δt, stat, roi)
        # Check indexing and pixel type *after* possible conversions.
        eltype(obj.stat) === T || argument_error(
            "invalid pixel type (expecting `$T`, got `$(eltype(obj.stat))`)")
        Base.has_offset_axes(obj.stat) && argument_error(
            "array of pixels must have 1-based indices")
        return obj
    end
end

function Base.push!(A::CalibrationData{T,N},
                    x::CalibrationDataStat{<:Real,N}) where {T<:AbstractFloat,N}
                    
    # Extract and check fields.
    cat = x.cat
    haskey(A.cat_index, cat) || argument_error(
        "category\"", cat, "\" does not exists in calibration data")
    Δt = x.Δt
    roi = x.roi
    Δt ≥ 0 || argument_error("exposure time must be nonnegative")
    Base.has_offset_axes(x.stat) && argument_error(
        "array of pixels must have 1-based indices")
    dims = size(x.stat)
    roi == DetectorAxes(A) || argument_error(
        "detector ROI settings must be identical for all calibration data")

    # Update statistics for given category and exposure time.
    key = (cat, Δt)
    if haskey(A.stat_index, key)
        index = A.stat_index[key]
        stat = A.stat[index]
        if storage(stat, 2) === A.null
            # Pushing one more sample will result in non-zero 2nd moment.
            # Allocate one for this sub-dataset.
            @assert nobs(stat) == 1
            A.stat[index] = IndependentStatistics(
                (storage(stat, 1), zeros(T, dims)), nobs(stat))
        end
    else
        # Create new instance of statistics (reusing `x` memory would be unexpected by user)
        push!(A.stat, IndependentStatistics((zeros(T,dims), zeros(T,dims)), 0))
        index = length(A.stat)
        A.stat_index[key] = index
    end
    merge!(A.stat[index], x.stat)
    return A
end


struct ImageStat{T<:Real,N}
    Δt::Float64                 # Exposure time.
    stat::OnlineStatistics{T,N} # Statistics.
    roi::DetectorAxes{N}  # Detector axes settings.
end
hduname(::Type{<:ImageStat}) = ("IMAGE-STAT", 1)
exposuretime(imgst::ImageStat) = imgst.Δt
StatsBase.nobs(imgst::ImageStat) = imgst.stat.n
DetectorAxes(imgst::ImageStat) = imgst.roi
