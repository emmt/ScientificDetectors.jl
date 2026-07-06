module TestingScientificDetectors

using Test, ScientificDetectors
using Aqua
using ScientificDetectors: offset, binning
using AstroFITS
using LinearAlgebra # to test other Arrays subtypes
using StructuredArrays # to test FastUniformArray
using Random
using LazyArtifacts
using StatsBase


@testset "Scientific Detectors" begin

@testset "Detector Axes" begin
    # Detector Axis.
    len, off, bin = 17, 3, 4
    axis = @inferred DetectorAxis(len; off=off, bin=bin)
    @test length(axis) == len
    @test offset(axis) == off
    @test binning(axis) == bin
    @test step(axis) == bin
    @test @inferred(DetectorAxis(axis)) === axis
    #
    axis = @inferred DetectorAxis(len)
    @test length(axis) == len
    @test offset(axis) == 0
    @test binning(axis) == 1
    @test step(axis) == 1
    @test @inferred(DetectorAxis(axis)) === axis
    #
    @test @inferred(DetectorAxis(Base.OneTo(7))) === @inferred DetectorAxis(7)
    @test @inferred(DetectorAxis(0x01:0x04)) === @inferred DetectorAxis(4)
    @test @inferred(DetectorAxis(2:1:5)) === @inferred DetectorAxis(4; off=1, step=1, bin=1)
    @test @inferred(DetectorAxis(3:2:5)) === @inferred DetectorAxis(2; off=2, step=2, bin=1)

    # Detector Axes.
    T = UInt8
    dims = (3, 4, 5)
    roi = @inferred DetectorAxes(dims)
    @test @inferred(DetectorAxes(dims...)) === roi
    @test @inferred(DetectorAxes{length(dims)}(dims)) === roi
    @test @inferred(DetectorAxes{length(dims)}(dims...)) === roi
    @test @inferred(DetectorAxes(map(DetectorAxis, dims))) === roi
    @test eltype(roi) === DetectorAxis
    @test length(roi) == length(dims)
    #
    @test size(roi) == dims
    @test_throws ArgumentError("out of range dimension index") size(roi, 0)
    @test size(roi, 2) == dims[2]
    @test size(roi, 1+length(dims)) == 1
    #
    @test axes(roi) == map(Base.OneTo, dims)
    @test_throws ArgumentError("out of range dimension index") axes(roi, 0)
    @test axes(roi, 2) == Base.OneTo(dims[2])
    @test axes(roi, 1+length(dims)) == Base.OneTo(1)
    #
    A = Array{T}(undef, dims)
    @test @inferred(DetectorAxes(A)) === roi
    @test @inferred(DetectorAxes{ndims(A)}(A)) === roi
    #
    t = @inferred Tuple(roi)
    @test @inferred(roi[1]) === t[1]
    @test @inferred(roi[2]) === t[2]
    @test @inferred(roi[3]) === t[3]
    @test @inferred(collect(roi)) == @inferred(collect(t))
end

@testset "ReducedCalibration constructors" begin

    #TODO some constructors are not tested yet, as the "more complex" ones in ReducedCalibration.jl


    # Inner constructor.
    (W,H) = dims = (32, 24)
    T = Float32
    roi = @inferred DetectorAxes(dims)
    A = @inferred ReducedCalibration{T,2,Array{Bool,2},Array{T,2}}(
        roi,
        ones(Float64, dims),
        Transpose(ones(Float64, dims[2], dims[1])), # test another subtype of AbstractArray
        ones(Float64, dims),
        ones(Float64, dims),
        ones(Float64, dims),
        [ones(Float64, dims), ones(Float64, dims),],
        ["TOTO", "TATA"],
        trues(dims),
        :zgσs;
        check = true)
    @test typeof(A) === ReducedCalibration{T,2,Array{Bool,2},Array{T,2}}


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
    for file in [ backs_paths ; flats_paths ; science_path ; goal_calib_data_path ;
                  goal_first_vpm_path ; goal_reduced_calib_data_path ; goal_reduced_science_path ]
        isfile(file) || error("Test set misses file: \"$file\".")
    end

    # basic info for tests
    typefloat = FitsFile(f -> f[1].data_eltype, science_path)
    (W, H) = FitsFile(f -> (f[1].data_size[1], f[1].data_size[2]), science_path)
    roi = DetectorAxes(W, H)
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
        goal_calib_data = readfits(CalibrationData, goal_calib_data_path)
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
    local vpm
    @testset "ReducedCalibration" begin
        Random.seed!(1234)

        local first_vpm
        @test_nowarn first_vpm = findbadpixels(calib_data)
        # we load the reference first_vpm file to compare
        goal_first_vpm = readfits(Array{Bool}, goal_first_vpm_path)
        @test first_vpm == goal_first_vpm

        reduced_calib_data = ReducedCalibration(calib_data; validpixels=first_vpm)
        @test reduced_calib_data isa ReducedCalibration
        @test_nowarn findbadpixels!(reduced_calib_data)

        # we load the reference ReducedCalibration file to compare
        goal_reduced_calib_data = readfits(ReducedCalibration, goal_reduced_calib_data_path)
        
        @test reduced_calib_data.roi == goal_reduced_calib_data.roi
        @test length(reduced_calib_data.src) == length(goal_reduced_calib_data.src)
        @test reduced_calib_data.algo == goal_reduced_calib_data.algo
       
        # compare vpm.
        @test count(reduced_calib_data.vpm .!= goal_reduced_calib_data.vpm) < 20
        
        # we only compare good pixels (by intersect the two vpm)
        vpm = reduced_calib_data.vpm .* goal_reduced_calib_data.vpm

        # comparing pixel to pixel, for detector characteristics and sources.
        @test all(.≈(reduced_calib_data.f[vpm], goal_reduced_calib_data.f[vpm]; rtol=0.05))
        @test all(.≈(reduced_calib_data.g[vpm], goal_reduced_calib_data.g[vpm]; atol=0.5))
        @test all(.≈(reduced_calib_data.z[vpm], goal_reduced_calib_data.z[vpm]; atol=5))
        @test all(.≈(reduced_calib_data.σ[vpm], goal_reduced_calib_data.σ[vpm]; atol=0.5))
        @test all(.≈(reduced_calib_data.σa[vpm], goal_reduced_calib_data.σa[vpm]; atol=0.25))
        for f in 1:length(reduced_calib_data.src)
            @test all(.≈(reduced_calib_data.s[f][vpm], goal_reduced_calib_data.s[f][vpm]
                         ; rtol=0.02, atol=1))
        end
        # comparing means and std
        @test ≈(mean(reduced_calib_data.f[vpm]), mean(goal_reduced_calib_data.f[vpm]); rtol=0.01)
        @test ≈(std(reduced_calib_data.f[vpm]), std(goal_reduced_calib_data.f[vpm]); rtol=0.01)
        @test ≈(mean(reduced_calib_data.g[vpm]), mean(goal_reduced_calib_data.g[vpm]); rtol=0.01)
        @test ≈(std(reduced_calib_data.g[vpm]), std(goal_reduced_calib_data.g[vpm]); rtol=0.01)
        @test ≈(mean(reduced_calib_data.z[vpm]), mean(goal_reduced_calib_data.z[vpm]); rtol=0.01)
        @test ≈(std(reduced_calib_data.z[vpm]), std(goal_reduced_calib_data.z[vpm]); rtol=0.01)
        @test ≈(mean(reduced_calib_data.σ[vpm]), mean(goal_reduced_calib_data.σ[vpm]); rtol=0.01)
        @test ≈(std(reduced_calib_data.σ[vpm]), std(goal_reduced_calib_data.σ[vpm]); rtol=0.01)
        @test ≈(mean(reduced_calib_data.σa[vpm]), mean(goal_reduced_calib_data.σa[vpm]); rtol=0.01)
        @test ≈(std(reduced_calib_data.σa[vpm]), std(goal_reduced_calib_data.σa[vpm]); rtol=0.01)
        for f in 1:length(reduced_calib_data.src)
            @test ≈(mean(reduced_calib_data.s[f][vpm]), mean(goal_reduced_calib_data.s[f][vpm]); rtol=0.01)
            @test ≈(std(reduced_calib_data.s[f][vpm]), std(goal_reduced_calib_data.s[f][vpm]); rtol=0.01)
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

        @test all(.≈(reduced_data[vpm], goal_reduced_data[vpm] ; rtol=0.05))
        @test all(.≈(weights[vpm],      goal_weights[vpm]      ; rtol=0.2))
        
        @test ≈(mean(reduced_data[vpm]), mean(goal_reduced_data[vpm]); rtol=0.01)
        @test ≈(std(reduced_data[vpm]),   std(goal_reduced_data[vpm]); rtol=0.01)
        @test ≈(mean(weights[vpm]), mean(goal_weights[vpm]); rtol=0.01)
        @test ≈(std(weights[vpm]),   std(goal_weights[vpm]); rtol=0.01)
    end
end


end # @testset "Scientific Detectors"

end # module
