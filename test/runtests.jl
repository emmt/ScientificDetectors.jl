module TestingScientificDetectors

using Test, ScientificDetectors
using Aqua
using ScientificDetectors: offset, binning
using AstroFITS
using LinearAlgebra # to test other Arrays subtypes
using StructuredArrays # to test FastUniformArray
using Random
using LazyArtifacts


@testset "runtests.jl" begin

@testset "DetectorAxis" begin
    # DetectorAxis
    len, off, bin = 17, 3, 4
    axis = DetectorAxis(len; off=off, bin=bin)
    @test length(axis) == len
    @test offset(axis) == off
    @test binning(axis) == bin
    axis = DetectorAxis(len)
    @test length(axis) == len
    @test offset(axis) == 0
    @test binning(axis) == 1

    # DetectorAxes
    T = UInt8
    dims = (3, 4, 5)
    roi = DetectorAxes(map(len -> DetectorAxis(len; off=0, bin=1), dims))
    @test size(roi) == dims
    @test_throws ErrorException size(roi, 0)
    @test size(roi, 2) == dims[2]
    @test size(roi, 1+length(dims)) == 1
    @test axes(roi) == map(Base.OneTo, dims)
    @test_throws ErrorException axes(roi, 0)
    @test axes(roi, 2) == Base.OneTo(dims[2])
    @test axes(roi, 1+length(dims)) == Base.OneTo(1)
    @test DetectorAxes(dims) == roi
    A = Array{T}(undef, dims)
    @test DetectorAxes(A) == roi
end

@testset "CalibrationData IO" begin
    # creating some test CalibrationData
    T = Float32
    (W, H) = (10, 10)
    roi = DetectorAxes(DetectorAxis(W; bin=2), DetectorAxis(H, off=3))
    dark_cat = CalibrationCategory("DARK", :(dark       ))
    flat_cat = CalibrationCategory("FLAT", :(dark + flat))
    calib = CalibrationData{T}(roi, [dark_cat, flat_cat])
    (Δt1, Δt2, Δt3) = (1.0, 2.0, 1.5)
    (nf1, nf2, nf3) = (5, 6, 13)
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf1), dark_cat.name, Δt1; roi))
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf2), flat_cat.name, Δt2; roi))
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf3), flat_cat.name, Δt3; roi))
    mktempdir() do dir
        filepath = joinpath(dir, "test.fits")
        FitsFile(filepath, "w!") do fitsfile
            @test_nowarn write(fitsfile, FitsHeader("TEST" => true), calib)
        end
        FitsFile(filepath) do fitsfile
            local calib2
            @test_nowarn calib2 = read(CalibrationData, fitsfile)
            @test calib.roi == calib2.roi
            @test calib.stat_index == calib2.stat_index
            @test calib.stat == calib2.stat
            @test calib.null == calib2.null
            @test calib.src_to_cat == calib2.src_to_cat
            @test calib.cat_index == calib2.cat_index
            @test calib.src_index == calib2.src_index
        end
    end
end

@testset "CalibrationData IO" begin
    # creating some test CalibrationData
    (W, H) = (10, 10)
    roi = DetectorAxes(DetectorAxis(W; bin=2), DetectorAxis(H, off=3))
    dark_cat = CalibrationCategory("DARK", :(dark       ))
    flat_cat = CalibrationCategory("FLAT", :(dark + flat))
    T = Float32
    calib = CalibrationData{T}(roi, [dark_cat, flat_cat])
    (Δt1, Δt2, Δt3) = (1.0, 2.0, 1.5)
    (nf1, nf2, nf3) = (5, 6, 13)
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf1), dark_cat.name, Δt1; roi))
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf2), flat_cat.name, Δt2; roi))
    push!(calib, CalibrationFrameSampler(rand(T, W, H, nf3), flat_cat.name, Δt3; roi))
    mktempdir() do dir
        filepath = joinpath(dir, "test.fits")
        FitsFile(filepath, "w!") do fitsfile
            @test_nowarn write(fitsfile, FitsHeader("TEST" => true), calib)
        end
        FitsFile(filepath) do fitsfile
            local calib2
            @test_nowarn calib2 = read(CalibrationData, fitsfile)
            @test calib.roi == calib2.roi
            @test calib.stat_index == calib2.stat_index
            @test calib.stat == calib2.stat
            @test calib.null == calib2.null
            @test calib.src_to_cat == calib2.src_to_cat
            @test calib.cat_index == calib2.cat_index
            @test calib.src_index == calib2.src_index
        end
    end
end

@testset "ReducedCalibration constructors" begin

    #TODO some constructors are not tested yet, as the "more complex" ones in ReducedCalibration.jl

    W = 2048
    H = 1024
    roi = DetectorAxes((1:W, 1:H))

    # inner
    @test ReducedCalibration{Float64,2,Array{Bool,2},Array{Float64,2}} == typeof(
            ReducedCalibration{Float64,2,Array{Bool,2},Array{Float64,2}}(
                roi,
                ones(Float64, W, H),
                Transpose(ones(Float64, H, W)), # test another subtype of AbstractArray
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                trues(W, H),
                :zgσs;
                check = true))

    # identity
    redcal = ReducedCalibration{Float64,2,Array{Bool,2},Array{Float64,2}}(
                roi,
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                trues(W, H),
                :zgσs;)
    @test redcal === ReducedCalibration(redcal)
    @test redcal === ReducedCalibration{Float64}(redcal)
    @test redcal === ReducedCalibration{Float64,2}(redcal)
    @test redcal === ReducedCalibration{Float64,2,Array{Bool,2}}(redcal)
    @test redcal === ReducedCalibration{Float64,2,Array{Bool,2},Array{Float64,2}}(redcal)

    # promote T (also test the default V and vpm constructors)
    redcal = ReducedCalibration(
        roi,
        ones(Float64, W, H),                        # f
        ones(Float32, W, H),                        # z
        ones(Float16, W, H),                        # g
        ones(Rational{Int}, W, H),                  # σ
        ones(Rational{Int}, W, H),                  # σa
        [ones(Int32, W, H), ones(BigFloat, W, H),], # s
        ["SRC1", "SRC2"])                           # src
    @test typeof(redcal) <: ReducedCalibration{BigFloat,2,<:FastUniformMatrix{Bool, true}}

    # T and V conversion
    redcal = ReducedCalibration{Float64,2,FastUniformMatrix{Bool,true},Array{Float64,2}}(
                roi,
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                ScientificDetectors.default_valid_pixels_map(roi),
                :zgσs;
                )
    @test ReducedCalibration{Float32,2,Array{Bool,2},Array{Float32,2}} == typeof(
        ReducedCalibration{Float32,2,Array{Bool,2},Array{Float32,2}}(redcal))

    # CalibrationData
    roy = DetectorAxes((1:2, 1:2))
    cat1 = CalibrationCategory("CAT1", :(cat1))
    caldat = CalibrationData{Float64}(roy, [cat1])
    redcal = ReducedCalibration(caldat)
    @test typeof(redcal) <: ReducedCalibration{Float64,2,<:FastUniformMatrix{Bool,true}}
end


@testset "Test on small IRDIS FITS files" begin

    DATA_DIR = artifact"SPHEREtestdata"

    flats_paths = [ joinpath(DATA_DIR, "flat_1s_db_h23.fits.gz"),
                    joinpath(DATA_DIR, "flat_3s_db_h23.fits.gz"),
                    joinpath(DATA_DIR, "flat_5s_db_h23.fits.gz")]

    backs_paths = [ joinpath(DATA_DIR, "back_1s_db_h23.fits.gz"),
                    joinpath(DATA_DIR, "back_8s_db_h23.fits.gz"),
                    joinpath(DATA_DIR, "back_96s_db_h23.fits.gz")]

    science_path = joinpath(DATA_DIR, "science_96s_db_h23.fits.gz")

    goal_calib_data_path = joinpath(DATA_DIR, "goal_calib_data.fits.gz")

    goal_first_vpm_path = joinpath(DATA_DIR, "goal_first_vpm.fits.gz")

    goal_reduced_calib_data_path = joinpath(DATA_DIR, "goal_reduced_calib_data.fits.gz")

    goal_reduced_science_path = joinpath(DATA_DIR, "goal_reduced_science_96s_db_h23.fits.gz")

    # ensuring test data files are present
    for file in [ backs_paths ; flats_paths ; goal_reduced_calib_data_path ;
                  science_path ; goal_reduced_science_path ]
        (isfile(file) && isreadable(file)) || error("Test set misses file: \"$file\".")
    end

    # basic info for tests
    typefloat = FitsFile(f -> f[1].data_eltype, science_path)
    roi = FitsFile(f -> DetectorAxes(f[1].data_size[1:2]), science_path)
    nbpixels = prod(size(roi))
    cats = [ CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]
    science_dit = FitsFile(f -> f[1]["EXPTIME"].float, science_path)

    # CalibrationData: use test data files
    local calib_data
    @testset "CalibrationData" begin
        Random.seed!(1234)
        @test_nowarn calib_data = CalibrationData{typefloat}(roi, cats)
        # push each calib file
        for (catname, filepaths) in [ ("BACK", backs_paths), ("FLAT", flats_paths) ]
            for filepath in filepaths
                FitsFile(filepath) do fitsfile
                    hdu = fitsfile[1]
                    realdit = typefloat(hdu["EXPTIME"].float)
                    cube = read(hdu, (:,:,:))
                    local sampler
                    @test_nowarn sampler = CalibrationFrameSampler(cube, catname, realdit; roi=roi)
                    @test_nowarn push!(calib_data, sampler)
                end
            end
        end
        
        # we load the reference file to compare
        goal_calib_data = read(CalibrationData, goal_calib_data_path)
        @test calib_data.src_index == goal_calib_data.src_index
        @test calib_data.cat_index == goal_calib_data.cat_index
        @test calib_data.src_to_cat == goal_calib_data.src_to_cat
        @test calib_data.stat_index == goal_calib_data.stat_index
        for i in eachindex(calib_data.stat, goal_calib_data.stat)
            @test all(calib_data.stat[i] .≈ goal_calib_data.stat[i])
        end
    end

    # ReducedCalibration
    local reduced_calib_data
    @testset "ReducedCalibration" begin
        Random.seed!(1234)

        local first_vpm
        @test_nowarn first_vpm = findbadpixels(calib_data)
        # we load the reference first_vpm file to compare
        goal_first_vpm = readfits(Array{Bool}, goal_first_vpm_path)
        @test all(first_vpm .== goal_first_vpm)

        reduced_calib_data = ReducedCalibration(calib_data; validpixels=first_vpm)
        @test reduced_calib_data isa ReducedCalibration
        @test_nowarn findbadpixels!(reduced_calib_data)

        # we load the reference ReducedCalibration file to compare
        goal_reduced_calib_data = read(ReducedCalibration, goal_reduced_calib_data_path)
        
        @test reduced_calib_data.roi == goal_reduced_calib_data.roi
        @test length(reduced_calib_data.src) == length(goal_reduced_calib_data.src)
        # compare vpm.
        @test all(reduced_calib_data.vpm .== goal_reduced_calib_data.vpm)
        # we only compare good pixels (by reference vpm)
        vpm = goal_reduced_calib_data.vpm
        # comparing pixel to pixel, for detector characteristics and sources.
        @test all(.≈(reduced_calib_data.f[vpm], goal_reduced_calib_data.f[vpm]; rtol=0.05))
        @test all(.≈(reduced_calib_data.g[vpm], goal_reduced_calib_data.g[vpm]; atol=0.05))
        @test all(.≈(reduced_calib_data.z[vpm], goal_reduced_calib_data.z[vpm]; atol=0.25))
        @test all(.≈(reduced_calib_data.σ[vpm], goal_reduced_calib_data.σ[vpm]; atol=0.25))
        for f in 1:length(reduced_calib_data.src)
            @test all(.≈(reduced_calib_data.s[f][vpm], goal_reduced_calib_data.s[f][vpm]
                         ; rtol=0.02, atol=1))
        end
    end

    # PreprocessingParameters
    local ppp
    @testset "PreprocessingParameters" begin
        Random.seed!(1234)
        @test_nowarn ppp = PreprocessingParameters(
            reduced_calib_data; flat="flat", bg="back", Δt=science_dit)
    end
    
    # process science
    @testset "process science" begin
        Random.seed!(1234)

        # load science input data file (49x45x1)
        data = readfits(Array{typefloat}, science_path)

        # reduced every frame
        reduced_data = similar(data)
        weights      = similar(data)
        for f in size(data,3)
            local w, rd
            @test_nowarn (w, rd) = process(ppp, view(data,:,:,f))
            reduced_data[:,:,f] .= rd
            weights[:,:,f] .= w
        end

        # load reduced science reference file to compare
        goal_reduced_data = readfits(Array{typefloat}, goal_reduced_science_path)
        goal_weights      = readfits(Array{typefloat}, goal_reduced_science_path; ext="weights")

        # we only compare good pixels (by reference vpm)
        vpm = (goal_weights .> 0)
        
        @test all(.≈(reduced_data[vpm], goal_reduced_data[vpm] ; rtol=0.02))
        @test all(.≈(weights[vpm],      goal_weights[vpm]      ; rtol=0.02))
    end
end

end # @testset "runtests.jl"
end # module
