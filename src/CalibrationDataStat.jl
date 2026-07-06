struct CalibrationDataStat{T<:Real,N}
    cat::String                 # Category.
    Δt::Float64                 # Exposure time.
    stat::IndependentStatistic{T,N,2} # Statistics.
    roi::DetectorAxes{N}  # Detector axes settings.
    
    function CalibrationDataStat{T,N}(cat::String,
                                      Δt::Real,
                                      stat::IndependentStatistic,
                                      roi::DetectorAxes{N}=DetectorAxes(size(stat))
                                      ) where {T<:Real,N}
        order(stat) < 2 && error("order must not be < 2")
        # converting to correct order if needed
        if order(stat) > 2
            stat = build_from_rawmoments(
                nobs(stat), (get_rawmoments(stat,1), get_rawmoments(stat,2)))
        end
        # converting to correct type if needed
        if !(stat isa IndependentStatistic{T})
            stat = build_from_rawmoments(
                nobs(stat), (map(T, get_rawmoments(stat,1)), map(T, get_rawmoments(stat,2))))
        end
        Δt ≥ 0 || argument_error("exposure time must be nonnegative")
        size(stat) == size(roi) || dimension_mismatch(
            "statistics and region of interest have different sizes")
        Base.has_offset_axes(stat) && argument_error("array of pixels must have 1-based indices")
        new{T,N}(cat, Δt, stat, roi)
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
    dims = size(mean(x.stat))
    TW = eltype(weights(x.stat))
    roi == DetectorAxes(A) || argument_error(
        "detector ROI settings must be identical for all calibration data")

    # Update statistics for given category and exposure time.
    key = (cat, T(Δt))
    if !haskey(A.stat_index, key)
        # Create new instance of statistics (reusing `x` memory would be unexpected by user)
        push!(A.stat, IndependentStatistic(T, 2, TW, dims))
        index = length(A.stat)
        A.stat_index[key] = index
    end
    index = A.stat_index[key]
    merge!(A.stat[index], x.stat)
    return A
end

