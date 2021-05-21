module DetectorStatistics

export
    SampleStatistics,
    exposuretime,
    numberofsamples,
    DetectorAxes

using ..ScientificDetectors
import ..ScientificDetectors: DetectorAxes, exposuretime

using EasyFITS
using EasyFITS: isprimary

using ArrayTools
using Statistics, StatsBase
using MultivariateOnlineStatistics

const OnlineStatistics{T,N} = IndependentStatistics{2,T,N,Array{T,N}}

struct SampleStatistics{T<:AbstractFloat,N}
    # Statistics.
    stat::OnlineStatistics{T,N}

    # Exposure time (in seconds).
    Δt::Float64

    # Detector geometry settings.
    roi::DetectorAxes{N}

    # This inner constructor is needed to avoid ambiguities and to check
    # compatibility of arguments.
    function SampleStatistics{T,N}(stat::OnlineStatistics{T,N},
                                   Δt::Real,
                                   roi::DetectorAxes{N}) where {T<:AbstractFloat,N}
        size(roi) == size(stat) || throw(DimensionMismatch(
            "statistics and region of interest have different dimensions"))
        new{T,N}(stat, Δt, roi)
    end
end

# Regular constructors (just provide missing type parameter).
function SampleStatistics{T}(A::OnlineStatistics{<:AbstractFloat,N},
                             Δt::Real,
                             roi::DetectorAxes{N} = DetectorAxes(size(A))
                             ) where {T<:AbstractFloat,N}
    SampleStatistics{T,N}(A, Δt, roi)
end
function SampleStatistics(A::OnlineStatistics{T,N},
                          Δt::Real,
                          roi::DetectorAxes{N} = DetectorAxes(size(A))
                          ) where {T<:AbstractFloat,N}
    SampleStatistics{T,N}(A, Δt, roi)
end

# Conversion constructors (return their input if possible, not a copy).
SampleStatistics(A::SampleStatistics) = A
SampleStatistics{T}(A::SampleStatistics{T}) where {T} = A
SampleStatistics{T,N}(A::SampleStatistics{T,N}) where {T,N} = A
SampleStatistics{T}(A::SampleStatistics{<:Any,N}) where {T,N} =
    SampleStatistics(
        OnlineStatistics{T,N}(
            map(x -> convert(Array{T,N}, x), storage(A.stat)), nobs(A.stat)),
        exposuretime(A),
        DetectorAxes(A))
SampleStatistics{T,N}(A::SampleStatistics{<:Any,N}) where {T,N} =
    SampleStatistics{T}(A)

# Constructors that read data from a file.
# FIXME: Add keyword `ext=...` to select the FITS extension.
SampleStatistics(path::AbstractString) = read(SampleStatistics, path)
SampleStatistics{T}(path::AbstractString) where {T} =
    read(SampleStatistics{T}, path)
SampleStatistics{T,N}(path::AbstractString) where {T,N} =
    read(SampleStatistics{T,N}, path)

# Constructors from given statistics.
function SampleStatistics(avg::AbstractArray{<:AbstractFloat,N},
                          std::AbstractArray{<:AbstractFloat,N},
                          nsamples::Integer,
                          Δt::Real,
                          roi::DetectorAxes{N} = DetectorAxes(avg);
                          corrected::Bool=true) where {N}
    T = promote_eltype(avg, std)
    return SampleStatistics{T,N}(avg, std, nsamples, Δt, roi; corrected=corrected)
end
function SampleStatistics{T}(avg::AbstractArray{<:AbstractFloat,N},
                             std::AbstractArray{<:AbstractFloat,N},
                             nsamples::Integer,
                             Δt::Real,
                             roi::DetectorAxes{N} = DetectorAxes(avg);
                             corrected::Bool=true) where {T<:AbstractFloat,N}
    return SampleStatistics{T,N}(avg, std, nsamples, Δt, roi; corrected=corrected)
end
function SampleStatistics{T,N}(avg::AbstractArray{<:AbstractFloat,N},
                               std::AbstractArray{<:AbstractFloat,N},
                               nsamples::Integer,
                               Δt::Real,
                               roi::DetectorAxes{N} = DetectorAxes(avg);
                               corrected::Bool=true) where {T<:AbstractFloat,N}
    # Check indexing.
    Base.has_offset_axes(avg, std) && error("arrays have non-standard indexing")
    axes(avg) == axes(std) || throw(DimensionMismatch(
        "arrays have different indices"))

    # Convert the standard deviation to the sum the of squared differences with
    # the empirical mean.  Then re-build an instance of OnlineStatistics.
    n = Int(nsamples)
    if corrected
        n -= 1
    end
    n ≥ 1 || throw(ArgumentError("not enough samples"))
    eta = T(n)
    s2 = Array{T,N}(undef, size(std))
    @inbounds @simd for i in eachindex(s2, std)
        s2[i] = eta*std[i]^2
    end
    s1 = convert(Array{T,N}, avg)
    stat = OnlineStatistics{T,N}((s1, s2), nsamples)

    return SampleStatistics{T,N}(stat, Δt, roi)
end

SampleStatistics(itr, Δt::Real) =
    SampleStatistics(Iterators.Stateful(itr), Δt)
function SampleStatistics(itr::Iterators.Stateful, Δt::Real)
    x1 = popfirst!(itr)
    T = float(eltype(x1))
    N = ndims(x1)
    return SampleStatistics{T,N}(x1, itr, Δt)
end

SampleStatistics{T}(itr, Δt::Real) where {T<:AbstractFloat} =
    SampleStatistics{T}(Iterators.Stateful(itr), Δt)
function SampleStatistics{T}(itr::Iterators.Stateful,
                             Δt::Real) where {T<:AbstractFloat}
    x1 = popfirst!(itr)
    N = ndims(x1)
    return SampleStatistics{T,N}(x1, itr, Δt)
end

SampleStatistics{T,N}(itr, Δt::Real) where {T<:AbstractFloat,N} =
    SampleStatistics{T,N}(Iterators.Stateful(itr), Δt)
function SampleStatistics{T,N}(itr::Iterators.Stateful,
                               Δt::Real) where {T<:AbstractFloat,N}
    x1 = popfirst!(itr)
    ndims(x1) == N || error("first sample does not have $N dimension(s)")
    return SampleStatistics{T,N}(x1, itr, Δt)
end

SampleStatistics(itr, Δt::Real, roi::DetectorAxes{N}) where {N} =
    SampleStatistics(Iterators.Stateful(itr), Δt, roi)
function SampleStatistics(itr::Iterators.Stateful, Δt::Real,
                          roi::DetectorAxes{N}) where {N}
    x1 = popfirst!(itr)
    ndims(x1) == N || error("first sample does not have $N dimension(s)")
    T = float(eltype(x1))
    return SampleStatistics{T,N}(x1, itr, Δt, roi)
end

SampleStatistics{T}(itr, Δt::Real, roi::DetectorAxes{N}) where {T<:AbstractFloat,N} =
    SampleStatistics{T}(Iterators.Stateful(itr), Δt, roi)
function SampleStatistics{T}(itr::Iterators.Stateful, Δt::Real,
                             roi::DetectorAxes{N}) where {T<:AbstractFloat,N}
    x1 = popfirst!(itr)
    ndims(x1) == N || error("first sample does not have $N dimension(s)")
    return SampleStatistics{T,N}(x1, itr, Δt, roi)
end

SampleStatistics{T,N}(itr, Δt::Real, roi::DetectorAxes{N}) where {T<:AbstractFloat,N} =
    SampleStatistics{T,N}(Iterators.Stateful(itr), Δt, roi)
function SampleStatistics{T,N}(itr::Iterators.Stateful, Δt::Real,
                               roi::DetectorAxes{N}) where {T<:AbstractFloat,N}
    dims = size(roi)
    stat = OnlineStatistics{T,N}((zeros(T, dims), zeros(T, dims)), 0)
    for x in itr
        push!(stat, x)
    end
    return SampleStatistics{T,N}(stat, Δt, roi)
end

# Pursue the construction with `x1` the first sample and `itr` an iterator over
# the other samples.
function SampleStatistics{T,N}(x1::AbstractArray{<:Real,N},
                               itr::Iterators.Stateful,
                               Δt::Real,
                               roi::DetectorAxes{N} = DetectorAxes(x)
                               ) where {T<:AbstractFloat,N}
    dims = size(roi)
    stat = OnlineStatistics{T,N}((zeros(T, dims), zeros(T, dims)), 0)
    push!(stat, x1)
    for x in itr
        push!(stat, x)
    end
    return SampleStatistics{T,N}(stat, Δt, roi)
end

# Extend methods for instances of SampleStatistics.
DetectorAxes(A::SampleStatistics) = A.roi
exposuretime(A::SampleStatistics) = A.Δt
StatsBase.nobs(A::SampleStatistics) = nobs(A.stat)
Statistics.mean(A::SampleStatistics) = mean(A.stat)
Statistics.std(A::SampleStatistics; corrected::Bool=true) =
    std(A.stat; corrected=corrected)
Statistics.var(A::SampleStatistics; corrected::Bool=true) =
    var(A.stat; corrected=corrected)
Base.ndims(A::SampleStatistics{T,N}) where {T,N} = N
Base.eltype(A::SampleStatistics{T,N}) where {T,N} = T
Base.length(A::SampleStatistics) = length(DetectorAxes(A))
Base.size(A::SampleStatistics) = size(DetectorAxes(A))
Base.size(A::SampleStatistics, i) = size(DetectorAxes(A), i)

Base.convert(::Type{T}, obj::SampleStatistics) where {T<:SampleStatistics} = T(obj)

Base.push!(A::SampleStatistics, x) = begin
    push!(A.stat, x)
    A
end

Broadcast.broadcasted(::Type{T}, obj::SampleStatistics) where {T<:AbstractFloat} =
    SampleStatistics{T}(obj)

Base.show(io::IO, obj::SampleStatistics{T,N}) where {T,N} = begin
    join(io, size(obj), "×")
    print(io, " PreprocessingParameters{$T,$N}: samples = ",
          nobs(obj), ", Δt = ", exposuretime(obj), " s")
end

end # module
