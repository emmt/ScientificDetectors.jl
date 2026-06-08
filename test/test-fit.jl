using Test, ScientificDetectors, Distributions

function sim(z, σ, g, Δt, src...)
    c = 0
    for s in src
        c += rand(Poisson(s * Δt))
    end
    x = (c * g) + rand(Normal(z,σ))
    round(x, RoundNearestTiesUp)
end

T = Float32
N = 1

src_dark = 1.0
src_flat = 100.0
src_lamp1 = 50.0
src_lamp2 = 88.0
src_lamp3 = 333.0
src_lamp4 = 4.44
g = 1.9
σ = 4
z = 10

roi = DetectorAxes(1:1)

nb_trials = 500

l_z = Vector{Float32}(undef, nb_trials)
l_g = Vector{Float32}(undef, nb_trials)
l_σ = Vector{Float32}(undef, nb_trials)
l_src_dark = Vector{Float32}(undef, nb_trials)
l_src_flat = Vector{Float32}(undef, nb_trials)
l_src_lamp1 = Vector{Float32}(undef, nb_trials)
l_src_lamp2 = Vector{Float32}(undef, nb_trials)
l_src_lamp3 = Vector{Float32}(undef, nb_trials)
l_src_lamp4 = Vector{Float32}(undef, nb_trials)

Threads.@threads for i in 1:nb_trials

    calib_data = CalibrationData{T}(roi, [
#            CalibrationCategory("FLAT", :(flat + dark)),
#            CalibrationCategory("DARK", :(dark)),
            CalibrationCategory("LAMP1", :(lamp1)),
#            CalibrationCategory("LAMP2", :(lamp2)),
#            CalibrationCategory("LAMP3", :(lamp3)),
#            CalibrationCategory("LAMP4", :(lamp4))
        ])

    for Δt in T[ 1.0, 3.0, 7.0, 10.0, 30.0, 50.0, 100.0, 1000.0 ]
        for f in 1:1000
#            push!(calib_data, CalibrationDataFrame{T,N}("FLAT", Δt, [sim(Δt, σ, z, src_flat, src_dark)] ))
#            push!(calib_data, CalibrationDataFrame{T,N}("DARK", Δt, [sim(Δt, σ, z, src_dark)] ))
            push!(calib_data, CalibrationDataFrame{T,N}("LAMP1", Δt, [sim(z,σ,g,Δt, src_lamp1)] ))
#            push!(calib_data, CalibrationDataFrame{T,N}("LAMP2", Δt, [sim(Δt, σ, z, src_lamp2)] ))
#            push!(calib_data, CalibrationDataFrame{T,N}("LAMP3", Δt, [sim(Δt, σ, z, src_lamp3)] ))
#            push!(calib_data, CalibrationDataFrame{T,N}("LAMP4", Δt, [sim(Δt, σ, z, src_lamp4)] ))
        end
    end

    r = ReducedCalibration(calib_data)
    
    l_z[i] = r.z[1]
    l_g[i] = r.g[1]
    l_σ[i] = r.σ[1]
#    push!(l_src_dark, r.s[findfirst(==("dark"),r.src)][1])
#    push!(l_src_flat, r.s[findfirst(==("flat"),r.src)][1])
    l_src_lamp1[i] = r.s[findfirst(==("lamp1"),r.src)][1]
#    l_src_lamp2[i] = r.s[findfirst(==("lamp2"),r.src)][1]
#    l_src_lamp3[i] = r.s[findfirst(==("lamp3"),r.src)][1]
#    l_src_lamp4[i] = r.s[findfirst(==("lamp4"),r.src)][1]
end

@show z, mean(l_z), std(l_z);
@show g, mean(l_g), std(l_g);
@show σ, mean(l_σ), std(l_σ);
#@show (src_dark * g), mean(l_src_dark), std(l_src_dark);
#@show (src_flat * g), mean(l_src_flat), std(l_src_flat);
@show (src_lamp1 * g), mean(l_src_lamp1), std(l_src_lamp1);
#@show (src_lamp2 * g), mean(l_src_lamp2), std(l_src_lamp2);
#@show (src_lamp3 * g), mean(l_src_lamp3), std(l_src_lamp3);
#@show (src_lamp4 * g), mean(l_src_lamp4), std(l_src_lamp4);


