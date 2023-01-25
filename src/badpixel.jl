
using StatsBase, Distributions



"""
    buildbadpixel!(C::ReducedCalibration{T,N}; threshold::Real=0.05, estimatedof = false,method::Symbol=:cov) where {T,N}
	
	buildbadpixel(A::CalibrationData{T,N} ; threshold::Real=0.05, estimatedof = false,method::Symbol=:cov) where {T,N}

Update the good pixel map according to all the quantities computed in `ReducedCalibration` or in  `CalibrationData.stat`

- `threshold` is the rejection threshold (default 0.05) 

- `estimatedof`  (default `false`), if `true`  the number of degree of freedom is estimated from the mean of the Χ^2 distribution. 

-  `method` select the method of Χ^2 computation: 
		- `:cov` use the empirical covariance matrix (default),
		- `:diag` use the empirical variance only,
		- `:robust` use robust estimates of empirical variance and mean using mad and median

"""
function buildbadpixel!(C::ReducedCalibration{T,N}; threshold::Real=0.05, estimatedof = false,method::Symbol=:cov) where {T,N}


	gpm = goodpixelmap(C)
	nvalid = count(gpm)

	nb_param = 4 + nsources(C)

	
	d = zeros(nvalid,nb_param) 

	# Co-log-likelihood.
	d[:,1] .= cologlikelihood(C)[gpm]
    # Zero-level 
	d[:,2] .= detectorbias(C)[gpm]
    # Detector gain 
	d[:,3] .= detectorgain(C)[gpm]
    # Standard deviation of the readout noise 
	d[:,4] .= detectornoise(C)[gpm]
	#sources
	@inbounds for i=1:nsources(C)
		d[:,4+i] .= sources(C,i)[gpm]
	end

	#m = [ median(d[:,i]) for i in 1:nb_param]
	#v = [ mad(d[:,i]) for i in 1:nb_param]
	Χ2 =Chi2(Val(method),d)
	if estimatedof
		dof = mean(Χ2)
	else
		dof = nb_param
	end
	C.gpm[gpm] .=  (Χ2 .< cquantile(Chisq(dof), threshold))
	return  C
end


function Chi2( ::Val{:cov},A::AbstractArray{T,2}) where {T<:AbstractFloat}
	m = mean(A, dims=1)
	t = (A .- m)
	U = cholesky(Symmetric(1/(size(A,1)-1)*(t'*t)); check=true).U
	return sum(abs2,t / U,dims=2)
end


function Chi2(::Val{:diag},A::AbstractArray{T,2}) where {T<:AbstractFloat}
	m = mean(A, dims=1)
	s = std(A, dims=1, mean=m)
	return sum(abs2,(A.-m) ./ s,dims=2)
end


function Chi2(::Val{:robust}, A::AbstractArray{T,2}) where {T<:AbstractFloat}
	m = median(A, dims=1)
	s = [ max.(1,mad(A[:,i],center=m[i])) for i in 1:size(A,2)]'
	return  sum(abs2,(A.-m) ./ s,dims=2)
end


function buildbadpixel(A::CalibrationData{T,N} ; threshold::Real=0.05, estimatedof = false,method::Symbol=:cov) where {T,N}

   # d = vcat(mean.( A.stat),var.( A.stat))

    nb_param = length(A.stat)
    numel = prod(size(A))
	gpm = trues(size(A))

    d = zeros(numel,nb_param)
    @inbounds for i in 1:nb_param 
        d[:,i] = mean(A.stat[i])[:]
    end

    Χ2 =Chi2(Val(method),d)
    if estimatedof
		dof = mean(Χ2)
	else
		dof = nb_param
	end

	gpm[:] .= (Χ2 .< cquantile(Chisq(nb_param), threshold))

	return gpm
end