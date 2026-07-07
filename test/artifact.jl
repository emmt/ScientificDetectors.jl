using Downloads, AstroFITS, ScientificDetectors, Random

const ARTIFACT_RANGE_X = 480:528 # non-square shape to detect erroneous axes inversion
const ARTIFACT_RANGE_Y = 482:526
const ARTIFACT_T = Float32

function download_artifact_raw_data()
    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T10:26:37.046",
        "back_1s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T11:16:20.263",
        "back_8s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T10:29:55.811",
        "back_96s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T12:01:47.854",
        "flat_1s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T12:07:09.316",
        "flat_3s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-06T12:13:13.151",
        "flat_5s_db_h23.fits.Z")

    @show Downloads.download(
        "http://dataportal.eso.org/dataportal_new/file/SPHER.2018-05-05T22:58:34.525",
        "science_96s_db_h23.fits.Z")

    function crop(filename, range_z)
        FitsFile(filename) do fitsfile
            hdu = fitsfile[1]
            header = FitsHeader(hdu)
            data = read(Array{ARTIFACT_T,3}, hdu, ARTIFACT_RANGE_X, ARTIFACT_RANGE_Y, range_z)
            filename_no_z = filename[1:end-2]
            writefits!(filename_no_z, filter(!is_structural, header), data)
            run(`gzip -f "$(filename_no_z)"`)
        end
        rm(filename)
    end

    crop("back_1s_db_h23.fits.Z", 1:10)
    crop("back_8s_db_h23.fits.Z", 1:10)
    crop("back_96s_db_h23.fits.Z", 1:10)

    crop("flat_1s_db_h23.fits.Z", 1:10)
    crop("flat_3s_db_h23.fits.Z", 1:10)
    crop("flat_5s_db_h23.fits.Z", 1:10)

    crop("science_96s_db_h23.fits.Z", 1:1)
end

function compute_artifact_data()
    # the results will be different if --check-bounds is enabled or not
    # the `pkg> test` command always set --check-bounds=yes
    # so we must always use that setting
    Base.JLOptions().check_bounds == 0 && error(
        "artifact data should be computed with julia option `--check-bounds=yes`.")

    science_Δt = 96.0
    roi = DetectorAxes(ARTIFACT_RANGE_X, ARTIFACT_RANGE_Y)
    cats = [ CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]

    Random.seed!(1234)
    calib_data = CalibrationData{ARTIFACT_T}(roi, cats)

    push!(calib_data,
        CalibrationFrameSampler(readfits("back_1s_db_h23.fits.gz"), "BACK", 1.0; roi))

    push!(calib_data,
        CalibrationFrameSampler(readfits("back_8s_db_h23.fits.gz"), "BACK", 8.0; roi))

    push!(calib_data,
        CalibrationFrameSampler(readfits("back_96s_db_h23.fits.gz"), "BACK", 96.0; roi))

    push!(calib_data,
        CalibrationFrameSampler(readfits("flat_1s_db_h23.fits.gz"), "FLAT", 1.0; roi))

    push!(calib_data,
        CalibrationFrameSampler(readfits("flat_3s_db_h23.fits.gz"), "FLAT", 3.0; roi))

    push!(calib_data,
        CalibrationFrameSampler(readfits("flat_5s_db_h23.fits.gz"), "FLAT", 5.0; roi))

    first_vpm = findbadpixels(calib_data)
    
    reduced_calib_data = ReducedCalibration(calib_data; validpixels=first_vpm)

    findbadpixels!(reduced_calib_data)

    ppp = PreprocessingParameters(
            reduced_calib_data; flat="flat", bg="back", Δt=science_Δt)

    data = readfits(Array{ARTIFACT_T}, "science_96s_db_h23.fits.gz")

    reduced_data = similar(data)
    weights      = similar(data)
    for f in size(data,3)
        (w, rd) = process(ppp, view(data,:,:,f))
        reduced_data[:,:,f] .= rd
        weights[:,:,f] .= w
    end 

    (calib_data, first_vpm, reduced_calib_data, ppp, reduced_data, weights)
end

function create_artifact_file()

    download_artifact_raw_data()

    (calib_data, first_vpm, reduced_calib_data, ppp, reduced_data, weights) = 
        compute_artifact_data()

    writefits!("goal_calib_data.fits", FitsHeader(), calib_data)
    writefits!("goal_first_vpm.fits", FitsHeader(), first_vpm)
    writefits!("goal_reduced_calib_data.fits", FitsHeader(), reduced_calib_data)
    writefits!("goal_reduced_science_96s_db_h23.fits",
               FitsHeader(), reduced_data,
               FitsHeader("EXTNAME" => "weights"), weights)

    run(`gzip -f goal_calib_data.fits`)
    run(`gzip -f goal_first_vpm.fits`)
    run(`gzip -f goal_reduced_calib_data.fits`)
    run(`gzip -f goal_reduced_science_96s_db_h23.fits`)

    run(`
        tar cf artifact.tar \
            back_1s_db_h23.fits.gz back_8s_db_h23.fits.gz back_96s_db_h23.fits.gz \
            flat_1s_db_h23.fits.gz flat_3s_db_h23.fits.gz flat_5s_db_h23.fits.gz \
            science_96s_db_h23.fits.gz \
            goal_calib_data.fits.gz \
            goal_first_vpm.fits.gz \
            goal_reduced_calib_data.fits.gz \
            goal_reduced_science_96s_db_h23.fits.gz`)
    run(`gzip -f artifact.tar`)

    rm("back_1s_db_h23.fits.gz")
    rm("back_8s_db_h23.fits.gz")
    rm("back_96s_db_h23.fits.gz")

    rm("flat_1s_db_h23.fits.gz")
    rm("flat_3s_db_h23.fits.gz")
    rm("flat_5s_db_h23.fits.gz")

    rm("science_96s_db_h23.fits.gz")
    
    rm("goal_calib_data.fits.gz")
    rm("goal_first_vpm.fits.gz")
    rm("goal_reduced_calib_data.fits.gz")
    rm("goal_reduced_science_96s_db_h23.fits.gz")
end
