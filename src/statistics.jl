module DetectorStatistics

export
    SampleStatistics,
    exposuretime,
    numberofsamples,
    DetectorAxes

using ..ScientificDetectors
using ..ScientificDetectors:
    dimension_mismatch
import ..ScientificDetectors:
    DetectorAxes,
    exposuretime

using EasyFITS
using EasyFITS: isprimary

using ArrayTools
using Statistics, StatsBase
using MultivariateOnlineStatistics

const OnlineStatistics{T,N} = IndependentStatistics{2,T,N,Array{T,N}}

"""
    A = SampleStatistics(stat, Δt, roi)

yields an object `A` collecting pixel-wise detector statistics `stat`, each
sample having the same exposure time `Δt` (in seconds) and corresponding to the
detector geometrical settings in `roi` (for "Region of Interest").

An instance may be built from the sample mean `avg` and standard deviation
`std` computed from `n` independent samples:

    A = SampleStatistics(avg, std, n, Δt, roi=DetectorAxes(avg); corrected=true)

unbiased standard deviation is assumed by default, use keyword
`corrected=false` otherwise.

Detector sample statistics can be read from a FITS file by one of:

    A = SampleStatistics(filename)
    A = read(SampleStatistics, filename)
    A = readfits(SampleStatistics, filename)

Detector sample statistics can also be written to a FITS file by:

    write(filename, A; overwrite=false)
    write!(filename, A)
    writefits(filename, A; overwrite=false)
    writefits!(filename, A)

Type parameters `T` (the floating-point type for computed statistics) and `N`
(the number of dimensions of the samples) may be specified in most constructor
calls:

    A = SampleStatistics{T}(...)
    A = SampleStatistics{T,N}(...)

The statistics can be updated on-line by calling:

    push!(A, x...) -> A

where each argument in `x...` is an idependent sample of detector measurements
under the same conditions (i.e exposure time and geometrical settings).

If `itr` is an iterable object producing measurements, then:

    merge!(A, itr) -> A

has the same effect as:

    foreach(x -> push!(A, x), itr)

It is also possible to merge the statistics from `B` into `A` by calling:

    merge!(A, B) -> A

provided both statistics are for the same exposure time and detector
geometrical settings (and the samples used in `A` and `B` are independent).

Basic methods:

    eltype(A)                         # the floating-point type of statistics
    length(A)                         # the number of pixels per sample
    ndims(A)                          # the number of dimensions of the samples
    size(A)                           # the dimensions of the samples
    size(A, i)                        # the i-th dimension of the samples
    DetectorAxes(A)                   # the detector geometry settings
    exposuretime(A)                   # the exposure time
    Statistics.mean(A)                # the mean of samples
    Statistics.std(A; corrected=true) # the standard deviation of samples
    Statistics.var(A; corrected=true) # the variance of samples
    StatsBase.nobs(A)                 # the number of independent samples

"""
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
        size(roi) == size(stat) || dimension_mismatch(
            "statistics and region of interest have different dimensions")
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
    axes(avg) == axes(std) || dimension_mismatch(
        "arrays have different indices")

    # Convert the standard deviation to the sum the of squared differences with
    # the empirical mean.  Then re-build an instance of OnlineStatistics.
    n = Int(nsamples)
    if corrected
        n -= 1
    end
    n ≥ 1 || argument_error("not enough samples")
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

Base.merge!(A::SampleStatistics, itr) = begin
    for x in itr
        push!(A, x)
    end
    A
end

Base.merge!(A::SampleStatistics, B::SampleStatistics) = begin
    size(B) == size(A) || dimension_mismatch(
        "statistics must have the same dimensions")
    DetectorAxes(B) == DetectorAxes(A) || dimension_mismatch(
        "samples must be for the same geometrical settings")
    exposuretime(B) == exposuretime(A) || dimension_mismatch(
        "samples must have the same exposure time")
    merge!(A.stat, B.stat)
    A
end

Broadcast.broadcasted(::Type{T}, obj::SampleStatistics) where {T<:AbstractFloat} =
    SampleStatistics{T}(obj)

Base.show(io::IO, obj::SampleStatistics{T,N}) where {T,N} = begin
    join(io, size(obj), "×")
    print(io, " SampleStatistics{$T,$N}: samples = ",
          nobs(obj), ", Δt = ", exposuretime(obj), " s")
end

end # module
