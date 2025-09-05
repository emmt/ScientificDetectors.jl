module TestingScientificDetectors

using Test, ScientificDetectors
using ScientificDetectors: offset, binning
using EasyFITS
using LinearAlgebra # to test other Arrays subtypes
using StructuredArrays # to test FastUniformArray

# for good mesure
using Random
Random.seed!(1234)

const DATA_DIR = joinpath(@__DIR__, "data/")

# list of small real calibration files (cropped 50x50 for space and time economy)
# they are used to test CalibrationData and ReducedCalibration structures
# an input science file (cropped 49x45) is also used to test reduction process
#
# there is also two "goal" files. they are used as reference, the test data is supposed to
# be very similar (floating point computation shall introduce small changes)

const BACKS_PATHS = map(x -> joinpath(DATA_DIR, x), ("back_1s.fits.gz",
                                                     "back_8s.fits.gz",
                                                     "back_96s.fits.gz"))

const FLATS_PATHS = map(x -> joinpath(DATA_DIR, x), ("flat_1s.fits.gz",
                                                    "flat_3s.fits.gz",
                                                    "flat_5s.fits.gz"))

const GOAL_REDUCED_CALIB_PATH   = joinpath(DATA_DIR, "goal_reduced_calib.fits.gz")

const INPUT_SCIENCE_PATH        = joinpath(DATA_DIR, "science.fits.gz")
const GOAL_REDUCED_SCIENCE_PATH = joinpath(DATA_DIR, "goal_reduced_science.fits.gz")

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
    roi = map(len -> DetectorAxis(len; off=0, bin=1), dims)
    @test size(roi) == dims
    @test_throws ErrorException size(roi, 0)
    @test size(roi, 2) == dims[2]
    @test size(roi, 1+length(dims)) == 1
    @test axes(roi) == map(Base.OneTo, dims)
    @test_throws ErrorException axes(roi, 0)
    @test axes(roi, 2) == Base.OneTo(dims[2])
    @test axes(roi, 1+length(dims)) == Base.OneTo(1)
    @test DetectorAxes(dims) === roi
    A = Array{T}(undef, dims)
    @test DetectorAxes(A) === roi
end

@testset "ReducedCalibration constructors" begin

    #TODO some constructors are not tested yet, as the "more complex" ones in ReducedCalibration.jl

    W = 2048
    H = 1024
    roi = DetectorAxes((1:W, 1:H))

    # inner
    @test ReducedCalibration{Float64,2,Array{Bool,2}} == typeof(
            ReducedCalibration{Float64,2,Array{Bool,2}}(
                roi,
                ones(Float64, W, H),
                Transpose(ones(Float64, H, W)), # test another subtype of AbstractArray
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                trues(W, H);
                check = true))

    # identity
    redcal = ReducedCalibration{Float64,2,Array{Bool,2}}(
                roi,
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                trues(W, H))
    @test redcal === ReducedCalibration(redcal)
    @test redcal === ReducedCalibration{Float64}(redcal)
    @test redcal === ReducedCalibration{Float64,2}(redcal)
    @test redcal === ReducedCalibration{Float64,2,Array{Bool,2}}(redcal)

    # promote T (also test the default V and vpm constructors)
    redcal = ReducedCalibration(
        roi,
        ones(Float64, W, H),                        # f
        ones(Float32, W, H),                        # z
        ones(Float16, W, H),                        # g
        ones(Rational{Int}, W, H),                  # σ
        [ones(Int32, W, H), ones(BigFloat, W, H),], # s
        ["SRC1", "SRC2"])                           # src
    @test typeof(redcal) <: ReducedCalibration{BigFloat,2,<:FastUniformMatrix{Bool, true}}

    # T and V conversion
    redcal = ReducedCalibration{Float64,2,FastUniformMatrix{Bool,true}}(
                roi,
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                ones(Float64, W, H),
                [ones(Float64, W, H), ones(Float64, W, H),],
                ["TOTO", "TATA"],
                ScientificDetectors.default_valid_pixels_map(roi))
    @test ReducedCalibration{Float32,2,Array{Bool,2}} == typeof(
        ReducedCalibration{Float32,2,Array{Bool,2}}(redcal))

    # CalibrationData
    roy = DetectorAxes((1:2, 1:2))
    cat1 = CalibrationCategory("CAT1", :(cat1))
    caldat = CalibrationData{Float64}(roy, [cat1])
    redcal = ReducedCalibration(caldat)
    @test typeof(redcal) <: ReducedCalibration{Float64,2,<:FastUniformMatrix{Bool,true}}
end


@testset "Test on small IRDIS FITS files" begin

    # ensuring test data files are present
    for file in [ BACKS_PATHS ; FLATS_PATHS ; GOAL_REDUCED_CALIB_PATH ;
                  INPUT_SCIENCE_PATH ; GOAL_REDUCED_SCIENCE_PATH ]
        isfile(file) || error("Test set misses file: \"$file\".")
    end

    # CalibrationData: use test data files
    local roi, cats, calibdata
    typefloat = FitsFile(f -> f[1].data_eltype, INPUT_SCIENCE_PATH)
    @testset "CalibrationData" begin
        @test_nowarn roi = FitsFile(f -> DetectorAxes(f[1].data_size[1:2]), INPUT_SCIENCE_PATH)
        @test_nowarn cats = [
            CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]
        @test_nowarn calibdata = CalibrationData{typefloat}(roi, cats)
        # push each calib file
        for (catname, filepaths) in [ ("BACK", BACKS_PATHS), ("FLAT", FLATS_PATHS) ]
            for filepath in filepaths
                FitsFile(filepath) do fitsfile
                    hdu = fitsfile[1]
                    realdit = typefloat(hdu["ESO DET SEQ1 REALDIT"].float)
                    cube = read(hdu, (:,:,:))
                    local sampler
                    @test_nowarn sampler = CalibrationFrameSampler(cube, catname, realdit; roi=roi)
                    @test_nowarn push!(calibdata, sampler)
                end
            end
        end
        src_index = calibdata.src_index
        cat_index = calibdata.cat_index
        @test length(src_index) == 2
        @test haskey(src_index, "back")
        @test haskey(src_index, "flat")
        @test length(cat_index) == 2
        @test haskey(cat_index, "BACK")
        @test haskey(cat_index, "FLAT")
        @test calibdata.src_to_cat[cat_index["BACK"], src_index["back"]] == 1
        @test calibdata.src_to_cat[cat_index["FLAT"], src_index["back"]] == 1
        @test calibdata.src_to_cat[cat_index["BACK"], src_index["flat"]] == 0
        @test calibdata.src_to_cat[cat_index["FLAT"], src_index["flat"]] == 1
    end

    # ReducedCalibration
    local reduced_calibdata
    @testset "ReducedCalibration" begin
        local firstvalidpixels
        @test_nowarn firstvalidpixels = findbadpixels(calibdata)
        reduced_calibdata = ReducedCalibration(calibdata;validpixels=firstvalidpixels)
        @test reduced_calibdata isa ReducedCalibration
        @test_nowarn findbadpixels!(reduced_calibdata)

        # we load the reference ReducedCalibration file to compare
        goal_reduced_calibdata = read(ReducedCalibration, GOAL_REDUCED_CALIB_PATH)
        
        @test reduced_calibdata.roi == goal_reduced_calibdata.roi
        @test length(reduced_calibdata.src) == length(goal_reduced_calibdata.src)
        # counting bad pixel differences between test and reference. few differences allowed.
        @test count(xor.(reduced_calibdata.vpm, goal_reduced_calibdata.vpm)) <= 16
        # we only compare good pixels (by reference vpm)
        vpm = goal_reduced_calibdata.vpm
        # comparing pixel to pixel, for detector characteristics and sources.
        @test all(.≈(reduced_calibdata.f[vpm], goal_reduced_calibdata.f[vpm]; atol=10))
        @test all(.≈(reduced_calibdata.g[vpm], goal_reduced_calibdata.g[vpm]; atol=0.05))
        @test all(.≈(reduced_calibdata.z[vpm], goal_reduced_calibdata.z[vpm]; atol=0.01))
        @test all(.≈(reduced_calibdata.σ[vpm], goal_reduced_calibdata.σ[vpm]; atol=0.01))
        for f in 1:length(reduced_calibdata.src)
            @test all(.≈(reduced_calibdata.s[f][vpm], goal_reduced_calibdata.s[f][vpm]; rtol=0.01))
        end
    end

    # reduced science
    @testset "PreprocessingParameters & reduce science" begin

        # load science input data file (49x45x1)
        science_dit = FitsFile(f->typefloat(f[1]["ESO DET SEQ1 REALDIT"].float), 
                          INPUT_SCIENCE_PATH)
        data = readfits(Array{typefloat}, INPUT_SCIENCE_PATH)

        local ppp, weights, reduced_data

        @test_nowarn ppp = PreprocessingParameters(
            reduced_calibdata; flat="flat", bg="back", Δt=science_dit)

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
        goal_reduced_data = readfits(Array{typefloat}, GOAL_REDUCED_SCIENCE_PATH)
        goal_weights      = readfits(Array{typefloat}, GOAL_REDUCED_SCIENCE_PATH; ext="weights")

        # we only compare good pixels (by reference vpm)
        vpm = (goal_weights .> 0)
        
        @test all(.≈(reduced_data[vpm], goal_reduced_data[vpm] ; rtol=0.01))
        @test all(.≈(weights[vpm],      goal_weights[vpm]      ; rtol=0.01))
    end
end

end # module
