module DetectorStatistics

export
    SampleStatistics,
    exposuretime,
    numberofsamples,
    regionofinterest

using ..ScientificDetectors
using ..ScientificDetectors: Sampler, offset, binning
import ..ScientificDetectors: regionofinterest, exposuretime, numberofsamples

using EasyFITS
using EasyFITS: isprimary

using ArrayTools
using Statistics

struct SampleStatistics{T<:AbstractFloat,N}
    # Sample mean.
    avg::Array{T,N}

    # Sample standard deviation.
    std::Array{T,N}

    # Number of samples.
    samples::Int

    # Exposure time (in seconds).
    Δt::Float64

    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::DetectorAxes{N}

    # Inner constructor provided to force using outer constructors.
    function SampleStatistics{T,N}(avg::Array{T,N},
                                   std::Array{T,N},
                                   samples::Integer,
                                   Δt::Real,
                                   roi::DetectorAxes{N}
                                   ) where {T<:AbstractFloat,N}
        @assert samples ≥ 2
        @assert isfinite(Δt) && Δt ≥ 0
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(avg) == dims
        @assert size(std) == dims
        return new{T,N}(avg, std, samples, Δt, roi)
    end
end

regionofinterest(obj::SampleStatistics) = obj.roi
exposuretime(obj::SampleStatistics) = obj.Δt
numberofsamples(obj::SampleStatistics) = obj.samples
Statistics.mean(obj::SampleStatistics) = obj.avg
Statistics.std(obj::SampleStatistics) = obj.std
Statistics.var(obj::SampleStatistics) = std(obj).^2
Base.ndims(obj::SampleStatistics{T,N}) where {T,N} = N
Base.eltype(obj::SampleStatistics{T,N}) where {T,N} = T
Base.length(obj::SampleStatistics) = length(regionofinterest(obj))
Base.size(obj::SampleStatistics) = size(regionofinterest(obj))
Base.size(obj::SampleStatistics, i) = size(regionofinterest(obj), i)

SampleStatistics(obj::SampleStatistics) = obj
SampleStatistics{T}(obj::SampleStatistics{T}) where {T} = obj
SampleStatistics{T,N}(obj::SampleStatistics{T,N}) where {T,N} = obj
SampleStatistics{T}(obj::SampleStatistics{<:Any,N}) where {T,N} =
    SampleStatistics{T,N}(obj.roi, obj.Δt, obj.samples,
                          convert(Array{T,N}, obj.avg),
                          convert(Array{T,N}, obj.std))
SampleStatistics{T,N}(obj::SampleStatistics{<:Any,N}) where {T,N} =
    SampleStatistics{T}(obj)

# FIXME: Add keyword `ext=...` to select the FITS extension.
SampleStatistics(path::AbstractString) = read(SampleStatistics, path)
SampleStatistics{T}(path::AbstractString) where {T} =
    read(SampleStatistics{T}, path)
SampleStatistics{T,N}(path::AbstractString) where {T,N} =
    read(SampleStatistics{T,N}, path)

Base.convert(::Type{T}, obj::SampleStatistics) where {T<:SampleStatistics} = T(obj)

Broadcast.broadcasted(::Type{T}, obj::SampleStatistics) where {T<:AbstractFloat} =
    SampleStatistics{T}(obj)

Base.show(io::IO, obj::SampleStatistics{T,N}) where {T,N} = begin
    join(io, size(obj), "×")
    print(io, " PreprocessingParameters{$T,$N}: samples = ",
          numberofsamples(obj), ", Δt = ", exposuretime(obj), " s")
end

regionofinterest(A::AbstractArray) = begin
    Base.has_offset_axes(A) && error("array has non-standrad indexing")
    map(DetectorAxis, size(A))
end

function SampleStatistics(avg::Array{T,N},
                          std::Array{T,N},
                          samples::Integer,
                          Δt::Real,
                          roi::DetectorAxes{N} = regionofinterest(avg)
                          ) where {T<:AbstractFloat,N}
    SampleStatistics{T,N}(avg, std, samples, Δt, roi)
end

function SampleStatistics(avg::AbstractArray{<:Real,N},
                          std::AbstractArray{<:Real,N},
                          samples::Integer,
                          Δt::Real,
                          roi::DetectorAxes{N} = regionofinterest(avg)
                          ) where {N}
    T = float(promote_type(eltype(avg), eltype(std)))
    SampleStatistics{T,N}(convert(Array{T,N}, avg),
                          convert(Array{T,N}, std), samples, Δt, roi)
end

function SampleStatistics{T}(avg::AbstractArray{<:Real,N},
                             std::AbstractArray{<:Real,N},
                             samples::Integer,
                             Δt::Real,
                             roi::DetectorAxes{N} = regionofinterest(avg)
                             ) where {T<:AbstractFloat,N}
    SampleStatistics{T,N}(convert(Array{T,N}, avg),
                          convert(Array{T,N}, std), samples, Δt, roi)
end

function SampleStatistics{T,N}(avg::AbstractArray{<:Real,N},
                               std::AbstractArray{<:Real,N},
                               samples::Integer,
                               Δt::Real,
                               roi::DetectorAxes{N} = regionofinterest(avg)
                               ) where {T<:AbstractFloat,N}
    SampleStatistics{T,N}(convert(Array{T,N}, avg),
                          convert(Array{T,N}, std), samples, Δt, roi)
end

SampleStatistics(dat::AbstractArray, Δt::Real, args...; kwds...) =
    SampleStatistics(Sampler(dat), Δt, args...; kwds...)

SampleStatistics{T}(dat::AbstractArray, Δt::Real, args...; kwds...) where {T<:AbstractFloat} =
    SampleStatistics{T}(Sampler(dat), Δt, args...; kwds...)

function SampleStatistics(dat::Union{AbstractVector{<:AbstractArray{T,N}},
                                     Tuple{Vararg{AbstractArray{T,N}}},
                                     Sampler{T,N}},
                          args...; kdws...) where {T<:Real,N}
    SampleStatistics{float(T)}(dat, args...; kdws...)
end

function SampleStatistics{T}(dat::Union{AbstractVector{<:AbstractArray{<:Real,N}},
                                        Tuple{Vararg{AbstractArray{<:Real,N}}},
                                        Sampler{T,N}},
                             Δt::Real,
                             roi::DetectorAxes{N} = regionofinterest(first(dat));
                             quick::Bool = false) where {T<:AbstractFloat,N}
    samples = length(dat)
    samples ≥ 2 || error("insufficient number of samples")
    dims = size(roi)
    S1 = zeros(T, dims)
    S2 = zeros(T, dims)
    q = T(1/samples)
    r = T(1/(samples - 1))
    if quick
        # Integerate statistics in a single pass.
        for A in dat
            Base.has_offset_axes(A) && error("data array has non-standard indexing")
            size(A) == dims ||
                throw(DimensionMismatch("all data arrays must have the same size"))
            _singlepass!(S1, S2, A)
        end
        @inbounds @simd for i in eachindex(S1, S2)
            s = S1[i]
            a = q*s
            S1[i] = a
            S2[i] = sqrt(max(S2[i] - a*s, zero(T))*r)
        end
    else
        # Integerate statistics in two passes.
        for A in dat
            Base.has_offset_axes(A) && error("data array has non-standard indexing")
            size(A) == dims ||
                throw(DimensionMismatch("all data arrays must have the same size"))
            _firstpass!(S1, A)
        end
        @inbounds @simd for i in eachindex(S1)
            S1[i] *= q
        end
        for A in dat
            _secondpass!(S2, A, S1)
        end
        @inbounds @simd for i in eachindex(S2)
            S2[i] = sqrt(r*S2[i])
        end
    end
    SampleStatistics(S1, S2, samples, Δt, roi)
end

function _singlepass!(S1::AbstractArray{T,N}, S2::AbstractArray{T,N},
                      A::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    @inbounds @simd for i in eachindex(S1, S2, A)
        a = T(A[i])
        S1[i] += a
        S2[i] += a*a
    end
    nothing
end

function _firstpass!(S1::AbstractArray{T,N},
                     A::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    @inbounds @simd for i in eachindex(S1, A)
        S1[i] += T(A[i])
    end
    nothing
end

function _secondpass!(S2::AbstractArray{T,N},
                      A::AbstractArray{<:Real,N},
                      avg::AbstractArray{T,N}) where {T<:AbstractFloat,N}
    @inbounds @simd for i in eachindex(S2, A)
        S2[i] += (T(A[i]) - avg[i])^2
    end
    nothing
end

end # module
