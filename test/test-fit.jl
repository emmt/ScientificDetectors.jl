using Test, ScientificDetectors, Distributions, ProgressMeter

T = Float32
N = 1

src_dark = 1.0
src_flat = 100.0
g = 1.9
σ = 4
z = 10

function sim(Δt, σ, z, src...)
    sum(s -> rand(Poisson(s * Δt)), src) * g + rand(Normal(z,σ))
end

roi = DetectorAxes(1:1)

l_z = Float32[]
l_g = Float32[]
l_σ = Float32[]
l_src_dark = Float32[]
l_src_flat = Float32[]

@showprogress for n in 1:1000

    calib_data = CalibrationData{T}(roi, [
            CalibrationCategory("FLAT", :(flat + dark)),
            CalibrationCategory("DARK", :(dark))
        ])

    for Δt in T[ 1.0, 3.0, 7.0, 10.0, 30.0, 100.0 ]
        for f in 1:1000
            push!(calib_data, CalibrationDataFrame{T,N}("FLAT", Δt, [sim(Δt, σ, z, src_flat, src_dark)] ))
            push!(calib_data, CalibrationDataFrame{T,N}("DARK", Δt, [sim(Δt, σ, z, src_dark)] ))
        end
    end

    r = ReducedCalibration(calib_data)
    
    push!(l_z, r.z[1])
    push!(l_g, r.g[1])
    push!(l_σ, r.σ[1])
    push!(l_src_dark, r.s[findfirst(==("dark"),r.src)][1])
    push!(l_src_flat, r.s[findfirst(==("flat"),r.src)][1])
end

@show z, mean(l_z), std(l_z);
@show g, mean(l_g), std(l_g);
@show σ, mean(l_σ), std(l_σ);
@show src_dark, mean(l_src_dark), std(l_src_dark);
@show src_flat, mean(l_src_flat), std(l_src_flat);

