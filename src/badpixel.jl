
using StatsBase, Distributions



"""
    buildbadpixel!(C::ReducedCalibration{T,N}; threshold::Real=0.05 ) where {T,N}

Update the bad pixel map according to all the quantities computed in `C`.  
"""
function buildbadpixel!(C::ReducedCalibration{T,N}; threshold::Real=0.05 ) where {T,N}

	nb_param = 4 + nsources(C)
	d = [ zeros(size(C)) for i in 1:nb_param]

	valid = badpixelmap(C)

	# Co-log-likelihood.
	d[1] = cologlikelihood(C)
    # Zero-level 
	d[2] = detectorbias(C)
    # Detector gain 
	d[3] = detectorgain(C)
    # Standard deviation of the readout noise 
	d[4] = detectornoise(C)
	#sources
	d[5:nb_param] .= sources(C)
	m = median.(d)
	v = mad.(d)

	whiten = (./).((.-).(d, m), max.(v,0.1 )) # broadcasted broadcast is rather unreadable
	
	C.bpm .= valid .& (sqrt.(sum(x -> abs2.(x),whiten)) .< cquantile.(Chisq(nb_param), threshold))

	return C
end