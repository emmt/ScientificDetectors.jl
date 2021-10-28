"""
    A = CalibrationDataFrame(cat, Δt, pxl; roi=DetectorAxes(pxl))

builds an instance of a calibration data frame with `cat` the category of the
calibration, `Δt` the exposure time (in seconds), and `pxl` the array of pixel
values.  Keyword `roi` may be used to specify a region of interest (that is the
geometric settings of the detector) other than the default.

The pixel type `T` and number `N` of dimensions can be specified as type
parameters:

    A = CalibrationDataFrame{T}(cat, Δt, pxl; kwds...)

"""
struct CalibrationDataFrame{T<:Real,N,A<:AbstractArray{T,N}}
    cat::String           # Category.
    Δt::Float64           # Exposure time.
    pxl::A                # Detector pixels.
    roi::DetectorAxes{N}  # Detector axes settings.

    # The inner constructor checks arguments and, for maximum flexibility, is
    # able to provide a default ROI and to convert most arguments (pxl, cat and
    # Δt).
    function CalibrationDataFrame{T,N,A}(cat::Category,
                                         Δt::Real,
                                         pxl::AbstractArray{<:Real,N};
                                         roi::DetectorAxes{N} = DetectorAxes(pxl)
                                         ) where {T<:Real,N,A<:AbstractArray{T,N}}
        Δt ≥ 0 || argument_error("exposure time must be nonnegative")
        size(pxl) == size(roi) || dimension_mismatch(
            "array of pixels and region of interest have different sizes")
        obj = new{T,N,A}(cat, Δt, pxl, roi)
        # Check indexing and pixel type *after* possible conversions.
        eltype(obj.pxl) === T || argument_error(
            "invalid pixel type (expecting `$T`, got `$(eltype(obj.pxl))`)")
        Base.has_offset_axes(obj.pxl) && argument_error(
            "array of pixels must have 1-based indices")
        return obj
    end
end

function CalibrationDataFrame(cat::Category,
                              Δt::Real,
                              pxl::AbstractArray{T,N};
                              kwds...) where {T<:Real,N}
    CalibrationDataFrame{T,N,typeof(pxl)}(cat, Δt, pxl; kwds...)
end

function CalibrationDataFrame{T}(cat::Category,
                                 Δt::Real,
                                 pxl::AbstractArray{<:Real,N};
                                 kwds...) where {T<:Real,N}
    # Do convert pixel type.
    CalibrationDataFrame{T,N,Array{T,N}}(cat, Δt, pxl; kwds...)
end

function CalibrationDataFrame{T}(cat::Category,
                                 Δt::Real,
                                 pxl::AbstractArray{T,N};
                                 kwds...) where {T<:Real,N}
    # No needs to convert pixel type.
    CalibrationDataFrame{T,N,typeof(pxl)}(cat, Δt, pxl; kwds...)
end

function CalibrationDataFrame{T,N}(cat::Category,
                                   Δt::Real,
                                   pxl::AbstractArray{<:Real,N};
                                   kwds...) where {T<:Real,N}
    # Numbers of dimensions match.  Call a simpler constructor.
    CalibrationDataFrame{T}(cat, Δt, pxl; kwds...)
end

# FIXME: make CalibrationDataFrame a sub-type of AbstractArray?

pixels(A::CalibrationDataFrame) = A.pxl
exposuretime(A::CalibrationDataFrame) = A.Δt
category(A::CalibrationDataFrame) = A.cat
DetectorAxes(A::CalibrationDataFrame) = A.roi
Base.size(A::CalibrationDataFrame) = size(pixels(A))
Base.ndims(A::CalibrationDataFrame) = ndims(typeof(A))
Base.ndims(::Type{<:CalibrationDataFrame{T,N}}) where {T,N} = N
Base.eltype(A::CalibrationDataFrame) = eltype(typeof(A))
Base.eltype(::Type{<:CalibrationDataFrame{T}}) where {T} = T
