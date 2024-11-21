using Test, ScientificDetectors, Distributions

function raw(; w, h, nbframes, flux, Δt, g, z, ωμ, ωσ, T)
    T[ let n = rand(Distributions.Poisson(flux * Δt))
           ω = rand(Distributions.Gaussian(ωμ, ωσ))
           round((n / g) + z + ω, RoundNearestTiesUp)
       end
       for x in 1:w, y in 1:h, f in 1:nbframes ]
end

@testset "test1" begin
    global calib, red
    
    w = 1
    h = 1
    nbframes = 20
    flux_flat = 1000.0 # [e/s] (not containing dark flux)
    flux_dark = 10.0   # [e/s]
    z = 5.0 # [ADU]
    g = 1.9 # [e/ADU]
    ωμ = 2.0 # [ADU]
    ωσ = 4.0 # [ADU]
    T = Float32
    
    raws_dark = map([1,5,10,20,40,60]) do Δt
        (Δt, raw(; w, h, nbframes, flux=flux_dark, Δt,  g, z, ωμ, ωσ, T))
    end
    raws_flat = map([1,5,10,20,40]) do Δt
        (Δt, raw(; w, h, nbframes, flux=flux_dark+flux_flat, Δt,  g, z, ωμ, ωσ, T))
    end

    roi = DetectorAxes(w,h)
    DARK_CAT = CalibrationCategory("DARK", :(dark       ))
    FLAT_CAT = CalibrationCategory("FLAT", :(dark + flat))
    calib = CalibrationData{T}(roi, [DARK_CAT, FLAT_CAT])
    
    foreach(raws_dark) do (Δt,r); push!(calib, CalibrationFrameSampler(r, DARK_CAT.name, Δt)) end
    foreach(raws_flat) do (Δt,r); push!(calib, CalibrationFrameSampler(r, FLAT_CAT.name, Δt)) end
    
    red = ReducedCalibration(calib)
    
    @test red.vpm[1] # [bool]
    @test isapprox(red.z[1], (z + ωμ); atol=0.5)  # [ADU]
    @test isapprox(red.g[1], g       ; atol=0.05) # [e/ADU]
    @test isapprox(red.σ[1], ωσ      ; atol=0.5)  # [ADU]
    @test isapprox(red.s[findfirst(==("dark"),red.src)][1], (flux_dark / g); rtol=0.05) # [ADU/s]
    @test isapprox(red.s[findfirst(==("flat"),red.src)][1], (flux_flat / g); rtol=0.05) # [ADU/s]
end


