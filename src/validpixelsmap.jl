using StatsBase, Distributions

const default_bad_pixels_threshold = 0.05

"""
    findbadpixels!(::ReducedCalibration; kwds...)

    findbadpixels(::CalibrationData; kwds...)

Update the valid pixels map according to all the quantities computed in `ReducedCalibration` or in  `CalibrationData.stat`

Possible keywords are:

- `threshold` is the rejection threshold (default $default_bad_pixels_threshold)

- `estimatedof`  (default `false`), if `true`  the number of degree of freedom is estimated from the mean of the Χ^2 distribution.

-  `method` selects the method of Χ^2 computation:
        - `:cov` use the empirical covariance matrix (default),
        - `:diag` use the empirical variance only,
        - `:robust` use robust estimates of empirical variance and mean using mad and median

For ::ReducedCalibration, a first :robust step is always executed on CoLogLikelihood,
because it often contains too huge values.

"""
function findbadpixels!(calib::ReducedCalibration{T,N};
                        threshold::Real = default_bad_pixels_threshold,
                        estimatedof::Bool = false,
                        method::Symbol = :cov) where {T,N}

    # first step with CoLogLikelihood
    # robust method to rule out the huge values that happen in CoLogLikeHood

    vpm = validpixelsmap(calib)
    L = reshape(cologlikelihood(calib)[vpm], :, 1)
    Χ2 = Chi2(Val(:robust),L)
    threshold_firststep = 0.0001
    validpixelsmap(calib)[vpm] .= (Χ2 .< cquantile(Chisq(1), threshold_firststep))


    # second step with the others columns
    # and the user chosen method

    vpm = validpixelsmap(calib)

    nb_param = 3 + nsources(calib)

    d = Matrix{T}(undef, count(vpm), nb_param)
    d[:,1] .= detectorbias(calib)[vpm]
    d[:,2] .= detectorgain(calib)[vpm]
    d[:,3] .= detectornoise(calib)[vpm]
    @inbounds for i = 1:nsources(calib) # every source
        d[:,3+i] .= sources(calib, i)[vpm]
    end

    Χ2 = Chi2(Val(method),d)

    dof = estimatedof ? mean(Χ2) : nb_param

    validpixelsmap(calib)[vpm] .= (Χ2 .< cquantile(Chisq(dof), threshold))

    return calib
end

"""Version with warning message about valid pixels map of type ::FastUniformArray"""
function findbadpixels!(
    calib::ReducedCalibration{T,N,FastUniformArray{Bool,N}}; kwds...) where {T,N}

    @warn string("Cannot modify pixels map of type ::FastUniformArray ; ",
                 "You can call ReducedCalibration{T,N,Array{Bool,N}}(::ReducedCalibration) ",
                 "to convert it to Array.")
    return calib
end

function Chi2(::Val{:cov}, A::AbstractArray{T,2}) where {T<:AbstractFloat}
    m = mean(A, dims=1)
    t = (A .- m)
    C = Symmetric(1/(size(A,1)-1) * (t'*t))
    F = cholesky(C; check=false)
    if issuccess(F)
        return sum(abs2, t / F.U, dims=2)
    else
        return sum(abs2, t * sqrt(Symmetric(pinv(C))), dims=2)
    end
end

function Chi2(::Val{:diag},A::AbstractArray{T,2}) where {T<:AbstractFloat}
    m = mean(A, dims=1)
    s = std(A, dims=1, mean=m)
    return sum(abs2, (A.-m) ./ s, dims=2)
end

function Chi2(::Val{:robust}, A::AbstractArray{T,2}) where {T<:AbstractFloat}
    m = median(A, dims=1)
    s = [ max.(1, mad(A[:,i], center=m[i], normalize=true)) for i in 1:size(A,2)]'
    return sum(abs2, (A.-m) ./ s, dims=2)
end

function findbadpixels(calib::CalibrationData{T,N};
                       threshold::Real = default_bad_pixels_threshold,
                       estimatedof = false,
                       method::Symbol = :cov,
                       init_vpm::AbstractArray{Bool}=FastUniformArray(true,size(calib))
                       ) where {T,N}

    nb_param = length(calib.stat)
    vpm = falses(size(calib))
    numel = count(init_vpm)

    d = zeros(numel, nb_param)
    @inbounds for i in 1:nb_param
        d[:,i] = mean(calib.stat[i])[init_vpm]
    end

    Χ2 = Chi2(Val(method),d)
    dof = estimatedof ? mean(Χ2) : nb_param

    vpm[init_vpm] .= (Χ2 .< cquantile(Chisq(nb_param), threshold))

    return vpm
end
