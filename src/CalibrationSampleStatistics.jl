struct CalibrationSampleStatistics{T<:Real,N}
    categoryname ::String
    samplestats  ::SampleStatistics{T,N}

    function CalibrationSampleStatistics{T,N}(categoryname ::AbstractString,
                                              samplestats  ::SampleStatistics{<:Real,N}
                                             ) where {T<:Real,N}
        samplestats.Δt ≥ 0 || argument_error("exposure time must be nonnegative")
        obj = new{T,N}(categoryname, samplestats)
        # Check indexing and pixel type *after* possible conversions.
        eltype(obj.samplestats) === T || argument_error(
            "invalid pixel type (expecting `$T`, got `$(eltype(obj.samplestats))`)")
        Base.has_offset_axes(obj.samplestats) && argument_error(
            "array of pixels statistics must have 1-based indices")
        return obj
    end
end

function Base.push!(A::CalibrationData{T,N},
                    x::CalibrationSampleStatistics{<:Real,N}) where {T<:AbstractFloat,N}
                    
    # Extract and check fields.
    cat = x.categoryname
    haskey(A.cat_index, cat) || argument_error(
        "category\"", cat, "\" does not exists in calibration data")
    Δt = exposuretime(x.samplestats)
    roi = DetectorAxes(x.samplestats)
    Δt ≥ 0 || argument_error("exposure time must be nonnegative")
    Base.has_offset_axes(x.samplestats) && argument_error(
        "array of pixels must have 1-based indices")
    dims = size(x.samplestats)
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
    merge!(A.stat[index], x.samplestats.stat)
    return A
end
