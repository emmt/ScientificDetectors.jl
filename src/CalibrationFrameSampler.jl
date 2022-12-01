"""
    S = CalibrationFrameSampler(arr, cat, Δt)

builds an iterator on `CalibrationDataFrame` from an array of frames `arr` and
corresponding category `cat` and exposure time (in seconds) `Δt`.  The pixel type
`T` can be specified as type parameter.

"""
struct CalibrationFrameSampler{T,N,Np1,A<:AbstractArray{T,Np1}}
    data::A
    inds::NTuple{N,Colon}
    cat::String           # Category.
    Δt::Float64           # Exposure time.
    roi::DetectorAxes{N}  # Detector axes settings.
    function CalibrationFrameSampler{T,N,Np1,A}(data::A,
                                                cat::String,
                                                Δt::Real;
                                                roi::DetectorAxes{N} = DetectorAxes(view(data, colons(N)..., 1))
                                                ) where {T,N,Np1,A<:AbstractArray{T,Np1}}

        Δt ≥ 0 || argument_error("exposure time must be nonnegative")
        Np1 == N + 1 || error("Np1 ≠ N + 1")
        Np1 ≥ 2 || error("insufficient number of dimensions")
        Base.has_offset_axes(data) && error(
            "data array has non-standard indexing")
        samples = size(data, Np1)
        samples ≥ 2 || error("insufficient number of samples")
        new{T,N,Np1,A}(data, colons(N),cat,Δt,roi)
    end
end

CalibrationFrameSampler(data::A,cat::String,Δt::Real) where {T,N,A<:AbstractArray{T,N}} =
        CalibrationFrameSampler{T,N-1,N,A}(data,cat,Δt)

CalibrationFrameSampler(data::A,cat::String,Δt::Real;roi::DetectorAxes{M}) where {T,N,M,A<:AbstractArray{T,N}} =
        CalibrationFrameSampler{T,M,N,A}(data,cat,Δt;roi=roi)


exposuretime(A::CalibrationFrameSampler) = A.Δt
category(A::CalibrationFrameSampler) = A.cat
DetectorAxes(A::CalibrationFrameSampler) = A.roi

StatsBase.nobs(A::CalibrationFrameSampler{T,N,Np1}) where {T,N,Np1} = size(A.data, Np1)

Base.eltype(A::CalibrationFrameSampler) = eltype(typeof(A))
# FIXME: be more specific
Base.eltype(::Type{<:CalibrationFrameSampler{T,N}}) where {T,N} =
    CalibrationDataFrame{T,N}

Base.IteratorEltype(A::CalibrationFrameSampler) = Base.IteratorEltype(typeof(A))
Base.IteratorEltype(::Type{<:CalibrationFrameSampler}) = Base.HasEltype()

Base.IteratorSize(A::CalibrationFrameSampler) = Base.IteratorSize(typeof(A))
Base.IteratorSize(::Type{<:CalibrationFrameSampler{T,N}}) where{T,N} =
    Base.HasShape{N}();

Base.ndims(A::CalibrationFrameSampler{T,N}) where {T,N} = N
Base.length(A::CalibrationFrameSampler) = nobs(A)
Base.size(A::CalibrationFrameSampler) = (length(A),)
Base.size(A::CalibrationFrameSampler{T,N}, i::Integer) where {T,N} =
    (i < 1 ? error("out of range dimension index") :
     i == 1 ? length(A) : 1)

Base.show(io::IO, obj::CalibrationFrameSampler{T,N}) where {T,N} = begin
    join(io, size(obj),"×")
    print(io, " CalibrationFrameSampler{$T,$N}: samples = ", nobs(obj))
end

Base.iterate(A::CalibrationFrameSampler, i = 1) =
    (1 ≤ i ≤ nobs(A) ? (CalibrationDataFrame(A.cat,A.Δt,view(A.data, A.inds..., i);roi=A.roi), i+1) : nothing)


function CalibrationData{T}(args::CalibrationFrameSampler...) where {T<:AbstractFloat}
    to_vector(x::AbstractVector) = x
    to_vector(x::Tuple) = collect(x)
    local roi::DetectorAxes, N::Int
    cat_index = Dict{String,Int}()
    args = to_vector(args)
    m = length(args) # number of categories
    m ≥ 1 || argument_error("there must be some categories")
    i = 0
    catarr  = Vector{CalibrationCategory}()
    for x in args
        cat = category(x)

        if !haskey(cat_index, cat)
            i +=1;
            cat_index[cat] = i;
            push!(catarr,CalibrationCategory(cat,Symbol(cat)));
        end
        if i == 1
            roi = DetectorAxes(x)
            N = length(roi)
        else
            roi == DetectorAxes(x) || argument_error(
                "detector ROI settings must be identical ",
                "for all calibration data")
        end
    end

    A = CalibrationData{T}(
        roi,                               # region of interest
        catarr
    )

    for x in args
        merge!(A, x)
    end
    return A
end


function Base.push!(A::CalibrationData{T,N}, args::CalibrationFrameSampler) where {T<:AbstractFloat,N}
    for x in args
        push!(A, x)
    end
    return A
end