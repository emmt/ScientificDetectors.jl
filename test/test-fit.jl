using Test, ScientificDetectors, Distributions, UnicodePlots, Plots

function compute_pixel_data(; z, g, σ, Δt, src_fluxes)
    electrons = 0
    for s in src_fluxes
        electrons += rand(Poisson(s * Δt))
    end
    readnoise = rand(Normal(0,σ))
    adus = round((electrons / g) + z + readnoise, RoundNearestTiesUp)
end

function new_calib_data(T, cats, srcs, files; z, g, σ)
    roi = DetectorAxes(1)
    calib_data_cats = CalibrationCategory[]
    for (cat_name, src_names) in cats
        src_expr = Meta.parse(join(src_names, " + "))
        push!(calib_data_cats, CalibrationCategory(cat_name, src_expr))
    end
    calib_data = CalibrationData{T}(roi, calib_data_cats)
    for (cat_name,Δt,nb_frames) in files
        src_names = cats[cat_name]
        src_fluxes = map(n -> srcs[n], src_names)
        file_data = Array{T}(undef, 1, nb_frames)
        for f in 1:nb_frames
            pixel_data = compute_pixel_data(; z, g, σ, Δt, src_fluxes)
            file_data[1,f] = pixel_data
        end
        push!(calib_data, CalibrationFrameSampler(file_data, cat_name, Δt; roi))
    end
    calib_data
end

function make_trials(nb_trials, T, categories, srcs, files; z, g, σ)
    result_z = zeros(T,nb_trials)
    result_g = zeros(T,nb_trials)
    result_σ = zeros(T,nb_trials)
    result_src_fluxes_adus = Dict(src_name => zeros(T,nb_trials) for (src_name,_) in srcs)
    Threads.@threads for i in 1:nb_trials
        calib_data = new_calib_data(T, categories, srcs, files; z, g, σ)
        r = ReducedCalibration(calib_data)
        result_z[i] = r.z[1]
        result_g[i] = r.g[1]
        result_σ[i] = r.σ[1]
        for (src_name,_) in srcs
            result_src_fluxes_adus[src_name][i] = r.s[findfirst(==(src_name),r.src)][1]
        end
    end
    (result_z, result_g, result_σ, result_src_fluxes_adus)
end

function benchmrk(nb_trials, T, categories, srcs, files; z, g, σ)
    (result_z, result_g, result_σ, result_src_fluxes_adus) = make_trials(
        nb_trials, T, categories, srcs, files; z, g, σ)
    if nb_trials > 1
        println("z:")
        show(UnicodePlots.histogram(result_z; xlabel=""))
        println()
        println("g:")
        show(UnicodePlots.histogram(result_g; xlabel=""))
        println()
        println("σ:")
        show(UnicodePlots.histogram(result_σ; xlabel=""))
        println()
    end
    println("z truth=$z, mean=$(mean(result_z)) std=$(std(result_z)) extrema=$(extrema(result_z))")
    println("g truth=$g, mean=$(mean(result_g)) std=$(std(result_g)) extrema=$(extrema(result_g))")
    println("σ truth=$σ, mean=$(mean(result_σ)) std=$(std(result_σ)) extrema=$(extrema(result_σ))")
    nothing
end

function test1()
    cats = Dict(
        "FLAT" => ["flat", "dark"],
        "DARK" => ["dark"],
    )

    srcs = Dict(
        "flat" => 100.0,
        "dark" => 10.0,
    )

    nb_frames_per_file = 50
    files = [
        ("FLAT", 1.0, nb_frames_per_file),
        ("FLAT", 5.0, nb_frames_per_file),
        ("FLAT", 10.0, nb_frames_per_file),
        ("DARK", 1.0, nb_frames_per_file),
        ("DARK", 5.0, nb_frames_per_file),
        ("DARK", 10.0, nb_frames_per_file),
    ]

    z =  10.10
    g = 1.9
    σ = 4.4

    T = Float32
    nb_trials = 1000

    benchmrk(nb_trials, T, cats, srcs, files; z, σ, g)
end


function test_gain_relative_to_flux()
    function subtest(flux_flat)
        cats = Dict(
            "FLAT" => ["flat", "dark"],
            "DARK" => ["dark"],
        )

        srcs = Dict(
            "flat" => flux_flat,
            "dark" => 1.0,
        )

        nb_frames_per_file = 50
        files = [
            ("FLAT", 1.0, nb_frames_per_file),
            ("FLAT", 5.0, nb_frames_per_file),
            ("FLAT", 10.0, nb_frames_per_file),
            ("DARK", 1.0, nb_frames_per_file),
            ("DARK", 5.0, nb_frames_per_file),
            ("DARK", 10.0, nb_frames_per_file),
        ]

        z =  10.10
        g = 1.9
        σ = 4.4

        T = Float32
        nb_trials = 1000

        (result_z, result_g, result_σ, result_src_fluxes_adus) = make_trials(
            nb_trials, T, cats, srcs, files; z, σ, g)
        
        (mean(result_g), std(result_g))
    end
    fluxes_flat = rand(100) .* 3000 .+ 10000
    means = []
    stds = []
    for flux_flat in fluxes_flat
        (m,s) = subtest(flux_flat)
        push!(means, m)
        push!(stds, s)
    end
    p = scatter(fluxes_flat, means; yerror=stds)
end



