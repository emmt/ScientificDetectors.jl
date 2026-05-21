struct SimpleCalibration{T<:AbstractFloat,N}
    # Dimensions, offsets and binning factors of the "Region Of Interest".
    roi::NTuple{N,DetectorAxis}

    # Exposure time (in seconds).
    Δt::Float64

    # Co-log-likelihood.
    f::Array{T,N}

    # Amplitude correction factor (in flux units per ADU):
    a::Array{T,N}

    # Bias correction (in ADU):
    b::Array{T,N}

    # Detector gain (in electrons per ADU):
    g::Array{T,N}

    # Standard deviation of the readout noise plus background (in ADU/frame):
    σ::Array{T,N}

    # Inner constructor provided to force using outer constructors.
    function SimpleCalibration{T,N}(roi::NTuple{N,DetectorAxis}, # TODO wrap in DetectorAxes
                                    Δt::Real,
                                    f::Array{T,N},
                                    a::Array{T,N},
                                    b::Array{T,N},
                                    g::Array{T,N},
                                    σ::Array{T,N}) where {T<:AbstractFloat,N}
        @assert isfinite(Δt) && Δt ≥ 0
        for i in 1:N
            @assert length(roi[i]) ≥ 1
            @assert offset(roi[i]) ≥ 0
            @assert binning(roi[i]) ≥ 1
        end
        dims = size(roi)
        @assert size(f) == dims
        @assert size(a) == dims
        @assert size(b) == dims
        @assert size(g) == dims
        @assert size(σ) == dims
        return new{T,N}(roi, Δt, f, a, b, g, σ)
    end
end

#
# Simple outer constructors for conversion.
#
SimpleCalibration(obj::SimpleCalibration) = obj
function SimpleCalibration(roi::NTuple{N,DetectorAxis}, # TODO DetectorAxes
                           Δt::Real,
                           f::AbstractArray{<:Real,N},
                           a::AbstractArray{<:Real,N},
                           b::AbstractArray{<:Real,N},
                           g::AbstractArray{<:Real,N},
                           σ::AbstractArray{<:Real,N}) where {N}
    T = float(promote_type(eltype(f), eltype(a), eltype(b), eltype(g), eltype(σ)))
    SimpleCalibration{T}(roi, Δt, f, a, b, g, σ)
end

SimpleCalibration{T}(obj::SimpleCalibration{T}) where {T} = obj
SimpleCalibration{T}(obj::SimpleCalibration{<:Any,N}) where {T<:AbstractFloat,N} =
    SimpleCalibration{T}(obj.roi, obj.Δt, obj.f, obj.a, obj.b, obj.g, obj.σ)
function SimpleCalibration{T}(roi::NTuple{N,DetectorAxis}, # TODO DetectorAxes
                              Δt::Real,
                              f::AbstractArray{<:Real,N},
                              a::AbstractArray{<:Real,N},
                              b::AbstractArray{<:Real,N},
                              g::AbstractArray{<:Real,N},
                              σ::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    # Call the inner constructor with all arguments of correct type.
    SimpleCalibration{T,N}(roi, Δt,
                           convert(Array{T,N}, f),
                           convert(Array{T,N}, a),
                           convert(Array{T,N}, b),
                           convert(Array{T,N}, g),
                           convert(Array{T,N}, σ))
end

SimpleCalibration{T,N}(obj::SimpleCalibration{T,N}) where {T,N} = obj
SimpleCalibration{T,N}(obj::SimpleCalibration{<:Any,N}) where {T,N} =
    SimpleCalibration{T}(obj)
function SimpleCalibration{T,N}(roi::NTuple{N,DetectorAxis}, # TODO DetectorAxes
                                Δt::Real,
                                f::AbstractArray{<:Real,N},
                                a::AbstractArray{<:Real,N},
                                b::AbstractArray{<:Real,N},
                                g::AbstractArray{<:Real,N},
                                σ::AbstractArray{<:Real,N}) where {T<:AbstractFloat,N}
    SimpleCalibration{T}(roi, Δt, f, a, b, g, σ)
end

#
# Getters.
#
DetectorAxes(obj::SimpleCalibration) = obj.roi
exposuretime(obj::SimpleCalibration) = obj.Δt

#
# Basic operations on SimpleCalibration structure.
#
Base.eltype(::SimpleCalibration{T}) where {T} = T
Base.size(obj::SimpleCalibration) = size(DetectorAxes(obj))
Base.size(obj::SimpleCalibration, i) = size(DetectorAxes(obj), i)
Base.length(obj::SimpleCalibration) = prod(size(obj))
Base.convert(::Type{T}, obj::SimpleCalibration) where {T<:SimpleCalibration} =
    T(obj)

function Base.show(io::IO, obj::SimpleCalibration{T,N}) where {T,N}
    print(io, "SimpleCalibration{$T,$N}: size = ")
    join(io, size(obj),"×")
    print(io, ", Δt = ", exposuretime(obj), " s")
end

# Allow for `T.(obj)` to work with `T` a floating-point type.
Broadcast.broadcasted(::Type{T}, obj::SimpleCalibration) where {T<:AbstractFloat} =
    SimpleCalibration{T}(obj)

"""
    SimpleCalibration{T}(ROI, Δt,
                         NumDark, AvgDark, VarDark,
                         NumLamp, AvgLamp, VarLamp,
                         AvgFlat)

yields reduced calibration data given sample means and variances of 3 kinds of
images: "dark" (or "bias") images, "lamp" images with a stable illumination
(although not necessarily uniform) and "flat" images with a uniform
illumination.  Arguments are as follows:

- `ROI` is a `N`-tuple of `DetectorAxis` describing the region of interest; # TODO DetectorAxes
- `Δt` is the exposure time (in seconds);
- `NumDark` is the number of averaged "dark" images;
- `AvgDark` is the sample mean of the "dark" images;
- `VarDark` is the sample variance of the "dark" images;
- `NumLamp` is the number of averaged "lamp" images;
- `AvgLamp` is the sample mean of the "lamp" images;
- `VarLamp` is the sample variance of the "lamp" images;
- `AvgFlat` is the sample mean of the "flat" images;

The sample variances are the maximum likelihood (i.e. biased) estimator of the
variances.  The sample mean and variance are computed as follows:

    avg = (1/N)*(x1 + x2 + ... + xN)
    var = (1/N)*((x1 - avg)^2 + (x2 - avg)^2 + ... + (xN - avg)^2)

Set keyword `unbiased` to `true` if the unbiased variances are provided, that
is computed as:

    var = (1/(N - 1))*((x1 - avg)^2 + (x2 - avg)^2 + ... + (xN - avg)^2)

See also [`ReducedCalibration`](@ref).

"""
function SimpleCalibration{T}(roi::NTuple{N,DetectorAxis},# TODO DetectorAxes
                              Δt::Real,
                              NumDark::Integer,
                              AvgDark::AbstractArray{<:Real,N},
                              VarDark::AbstractArray{<:Real,N},
                              NumLamp::Integer,
                              AvgLamp::AbstractArray{<:Real,N},
                              VarLamp::AbstractArray{<:Real,N},
                              AvgFlat::AbstractArray{<:Real,N};
                              optimal::Bool = false,
                              unbiased::Bool=false,
                              umin::Real=1e-20) where {T<:AbstractFloat,N}
    # Check arguments.
    @assert isfinite(Δt) && Δt > 0
    @assert isfinite(umin) && umin > 0
    dims = size(roi)
    @assert !Base.has_offset_axes(AvgDark, VarDark, AvgLamp, VarLamp, AvgFlat)
    @assert NumDark > 1
    @assert size(AvgDark) == dims
    @assert size(VarDark) == dims
    @assert NumLamp > 1
    @assert size(AvgLamp) == dims
    @assert size(VarLamp) == dims
    @assert size(AvgFlat) == dims

    # Local variables fopr computations.
    Ndark = Float64(NumDark)
    Nlamp = Float64(NumLamp)
    local Mdark::Float64, Vdark::Float64
    local Mlamp::Float64, Vlamp::Float64
    local Mflat::Float64

    # Allocate arrays for the result.
    f = Array{T}(undef, dims)
    a = Array{T}(undef, dims)
    b = Array{T}(undef, dims)
    g = Array{T}(undef, dims)
    σ = Array{T}(undef, dims)
    obj = SimpleCalibration{T,N}(roi, Δt, f, a, b, g, σ)

    # Perform a first pass to check the values and initialize parameters to
    # their sub-optimal estimators.
    checkvalues(Mdark, Vdark, Mlamp, Vlamp, Mflat) =
        ((isfinite(Mdark) & isfinite(Vdark) &
          isfinite(Mlamp) & isfinite(Vlamp) &
          isfinite(Mflat)) && 0 < Vdark < Vlamp &&
         Mdark < min(Mlamp, Mflat))

    # Variance correction factors for unbiased estimators.
    γdark = (unbiased ? one(Float64) : Ndark/(Ndark - 1))
    γlamp = (unbiased ? one(Float64) : Nlamp/(Nlamp - 1))

    for j in eachindex(f, a, b, g, σ, AvgDark, VarDark, AvgLamp, VarLamp)
        Mdark = Float64(AvgDark[j])
        Vdark = Float64(VarDark[j])*γdark
        Mlamp = Float64(AvgLamp[j])
        Vlamp = Float64(VarLamp[j])*γlamp
        Mflat = Float64(AvgFlat[j])
        if checkvalues(Mdark, Vdark, Mlamp, Vlamp, Mflat)
            f[j] = (Ndark*(1 + log(Vdark)) - 1) + (Nlamp*(1 + log(Vlamp)) - 1)
            a[j] = 1/(Mflat - Mdark)
            b[j] = Mdark
            g[j] = (Mlamp - Mdark)/(Vlamp - Vdark)
            σ[j] = sqrt(Vdark)
        else
            f[j] = Inf
            a[j] = 0
            b[j] = 0
            g[j] = 0
            σ[j] = Inf
        end
    end

    # Unless sub-optimal estimators have been requested, perform a second pass
    # to obtain optimal estimators.
    if optimal == false
        return obj
    end

    # Allocate workspace.  Computations are done in double precision.
    var = Vector{Float64}(undef, 2)
    #grd = Vector{Float64}(undef, 2)
    lim = Vector{Float64}(undef, 2)

    # Workspace to store the best solution so far.
    best = Vector{Float64}(undef, 5)

    # Objective function to minimize (as a closure).
    function fg!(var::Vector{Float64}, grd::Vector{Float64})
        local f, b, g, x, y
        # Extract parameters.
        @assert length(var) == length(grd) == 2
        x = var[1]
        y = var[2]

        # Weights.
        Wdark = Ndark/x
        Wlamp = Nlamp/(x + y)

        # Best bias.
        b = (Wdark*Mdark + Wlamp*(Mlamp - y))/(Wdark + Wlamp)

        # Residuals.
        Rdark = Mdark - b
        Rlamp = Mlamp - (b + y)

        # Best gain.
        g = (Ndark + Nlamp)/(Wdark*(Vdark + Rdark^2) + Wlamp*(Vlamp + Rlamp^2))

        # Gradient.
        gtemp = Wlamp*(1  - g*(Vlamp + Rlamp^2)/(x + y))
        grd[1] = gtemp + Wdark*(1  - g*(Vdark + Rdark^2)/x)
        grd[2] = gtemp - 2*Wlamp*g*Rlamp

        # Objective function.
        f = Ndark*(1 + log(x/g)) + Nlamp*(1 + log((x + y)/g))
        if f < best[1]
            best[1] = f
            best[2] = b
            best[3] = g
            best[4] = x
            best[5] = y
        end
        return f
    end

    # Variance correction factors for max. likelihood estimators.
    γdark = (unbiased ? (Ndark - 1)/Ndark : one(Float64))
    γlamp = (unbiased ? (Nlamp - 1)/Nlamp : one(Float64))

    for j in eachindex(f, a, b, g, σ, AvgDark, VarDark, AvgLamp, VarLamp)
        Mdark = Float64(AvgDark[j])
        Vdark = Float64(VarDark[j])*γdark
        Mlamp = Float64(AvgLamp[j])
        Vlamp = Float64(VarLamp[j])*γlamp
        Mflat = Float64(AvgFlat[j])
        if a[j] > 0
            best[1] = Inf
            best[2] = NaN
            best[3] = NaN
            best[4] = NaN
            best[5] = NaN
            # Initial solution:
            #     b  = Mdark
            #     g  = (Mlamp - Mdark)/(Vlamp - Vdark)
            #     x  = g*Vdark
            #     y  = Mlamp - Mdark
            var[1] = (Mlamp - Mdark)/(Vlamp/Vdark - 1) # initial x
            var[2] = Mlamp - Mdark                     # initial y
            lim[1] = var[1]/100
            lim[2] = 0

            vmlmb!(fg!, var, mem=2, lower=lim)
            if Mflat > best[2]
                f[j] = best[1]
                a[j] = 1/(Mflat - best[2])
                b[j] = best[2]
                g[j] = best[3]
                σ[j] = sqrt(best[4]/g[j])
            end
        end
    end

    return obj
end
