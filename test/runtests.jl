module TestingScientificDetectors

using Test, ScientificDetectors
using ScientificDetectors: offset, binning
using EasyFITS
using LinearAlgebra # to test other Arrays subtypes
using StructuredArrays # to test FastUniformArray

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

    # !! update this list if you modify the data test files !!
    backs_filepaths = map(f -> "test/data/" * f,
        ["back_1s.fits.gz", "back_8s.fits.gz", "back_96s.fits.gz"])
    flats_filepaths = map(f -> "test/data/" * f,
        ["flat_1s.fits.gz", "flat_3s.fits.gz", "flat_5s.fits.gz"])
    science_filepath = "test/data/science_96s.fits.gz"
    goal_reduced_calibdata_filepath = "test/data/goal_reduced_calibdata.fits.gz"
    goal_reduced_science_filepath   = "test/data/goal_reduced_science_96s.fits.gz"

    # ensuring resources files are present
    for file in [ backs_filepaths ; flats_filepaths ; science_filepath ;
                  goal_reduced_calibdata_filepath ; goal_reduced_calibdata_filepath ]
        isfile(file) || error("Test set misses file: \"$file\".")
    end

    # CalibrationData
    local roi, cats, calibdata
    typefloat = FitsFile(f -> f[1].data_eltype, science_filepath)
    @testset "CalibrationData" begin
        @test_nowarn roi = FitsFile(f -> DetectorAxes(f[1].data_size[1:2]), science_filepath)
        @test_nowarn cats = [
            CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]
        @test_nowarn calibdata = CalibrationData{typefloat}(roi, cats)
        for (catname, filepaths) in [ ("BACK", backs_filepaths), ("FLAT", flats_filepaths) ]
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
        #TODO: more tests when pull request for CalibrationData IO will be merged
    end

    # ReducedCalibration
    local reduced_calibdata
    @testset "ReducedCalibration" begin
        local firstvalidpixels
        @test_nowarn firstvalidpixels = findbadpixels(calibdata)
        reduced_calibdata = ReducedCalibration(calibdata;validpixels=firstvalidpixels)
        @test reduced_calibdata isa ReducedCalibration
        @test_nowarn findbadpixels!(reduced_calibdata)

        goal_reduced_calibdata = read(ReducedCalibration, goal_reduced_calibdata_filepath)
        @test reduced_calibdata.roi == goal_reduced_calibdata.roi
        @test count(xor.(reduced_calibdata.vpm, goal_reduced_calibdata.vpm)) <= 16

        @test reduced_calibdata.f ≈ goal_reduced_calibdata.f
        @test reduced_calibdata.g ≈ goal_reduced_calibdata.g
        @test reduced_calibdata.z ≈ goal_reduced_calibdata.z
        @test reduced_calibdata.σ ≈ goal_reduced_calibdata.σ
        @test length(reduced_calibdata.src) == length(goal_reduced_calibdata.src)
        for f in 1:length(reduced_calibdata.src)
            @test reduced_calibdata.s[f] ≈ goal_reduced_calibdata.s[f]
        end
    end

    # reduced science
    # we assume that science has NAXIS3 == 1
    @testset "PreprocessingParameters & reduce science" begin
        science_dit = FitsFile(f->typefloat(f[1]["ESO DET SEQ1 REALDIT"].float), science_filepath)
        science_matrix       = readfits(Matrix{typefloat}, science_filepath, :,:,1)
        goal_reduced_science = readfits(goal_reduced_science_filepath)
        local ppp, weights, reduced_data
        @test_nowarn ppp = PreprocessingParameters(
            reduced_calibdata; flat="flat", bg="back", Δt=science_dit)
        @test_nowarn (weights, reduced_data) = process(ppp, science_matrix)
        @test reduced_data ≈ goal_reduced_science[:,:,1]
        @test weights      ≈ goal_reduced_science[:,:,2]
    end

    # YAML config
    @testset "CalibrationDataFromYAML" begin
        local calibdatayaml
        @test_nowarn calibdatayaml= CalibrationData{Float32}(
            "test/data/config.yaml"; basedir="test/data/")
        src_index = calibdatayaml.src_index
        cat_index = calibdatayaml.cat_index
        @test length(src_index) == 2
        @test haskey(src_index, "back")
        @test haskey(src_index, "flat")
        @test length(cat_index) == 2
        @test haskey(cat_index, "BACK")
        @test haskey(cat_index, "FLAT")
        @test calibdatayaml.src_to_cat[cat_index["BACK"], src_index["back"]] == 1
        @test calibdatayaml.src_to_cat[cat_index["FLAT"], src_index["back"]] == 1
        @test calibdatayaml.src_to_cat[cat_index["BACK"], src_index["flat"]] == 0
        @test calibdatayaml.src_to_cat[cat_index["FLAT"], src_index["flat"]] == 1
        #TODO: more tests when pull request for CalibrationData IO will be merged
    end
end

end # module
