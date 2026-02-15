using Downloads, AstroFITS, ScientificDetectors

"""
This function should be runned from the test/data/ folder since it will download
the data files inside pwd()
"""
function generate_data_files(
    ; range_x = 480:528, # non-square shape to detect erroneous axes inversion
      range_y = 482:526,
      range_z = 1:10,
      backs_to_download = [
          ("SPHER.2018-05-06T10:26:37.046", "back_1s.fits.Z"),
          ("SPHER.2018-05-06T11:16:20.263", "back_8s.fits.Z"),
          ("SPHER.2018-05-06T10:29:55.811", "back_96s.fits.Z")],
      flats_to_download = [
          ("SPHER.2018-05-06T12:01:47.854", "flat_1s.fits.Z"),
          ("SPHER.2018-05-06T12:07:09.316", "flat_3s.fits.Z"),
          ("SPHER.2018-05-06T12:13:13.151", "flat_5s.fits.Z")],
      science_to_download = ("SPHER.2018-05-05T22:58:34.525", "science_96s.fits.Z"),
      goal_reduced_calibdata_filename = "goal_reduced_calibdata.fits",
      goal_reduced_science_filename   = "goal_reduced_science_96s.fits",
      typefloat = Float32
)
    range_z_science = 1:1 # only one frame for science. or update code for reducing science

    files_to_download = [backs_to_download ; flats_to_download ; science_to_download]
    l = length(files_to_download)

    # Downloading
    ESO_DOWNLOAD_URL ::String = "http://dataportal.eso.org/dataportal_new/file/"
    for (i, (dpid, filename)) in enumerate(files_to_download)
        @info "Downloading file $i/$l.."
        Downloads.download(ESO_DOWNLOAD_URL * dpid, filename)
    end

    local science_dit

    # Croping and compressing
    for (i,filename) in enumerate(map(p->p[2], files_to_download))
        @info "Croping and compressing file $i/$l.."
        local header, data
        FitsFile(filename) do fitsfile
            hdu = fitsfile[1]
            header = FitsHeader(hdu)
            header["NAXIS1"] = length(range_x)
            header["NAXIS2"] = length(range_y)
            if filename == science_to_download[2]
                header["NAXIS3"] = length(range_z_science)
                data = read(Array{typefloat,3}, hdu, range_x, range_y, range_z_science)
                science_dit = typefloat(header["ESO DET SEQ1 REALDIT"].float)
            else
                header["NAXIS3"] = length(range_z)
                data = read(Array{typefloat,3}, hdu, range_x, range_y, range_z)
            end
        end
        run(`rm "$(filename)"`)
        filename = chop(filename; tail=2) # without .Z
        header2 = filter(!is_structural, header)

        writefits!(filename, header2, data)

        run(`gzip -f "$(filename)"`) # creates the .fits.gz file and auto delete the .fits file
    end

    # Applying Scientific Detectors
    @info "Applying Scientific Detectors"

    roi = DetectorAxes(1:length(range_x), 1:length(range_y))
    cats = [ CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]

    # CalibrationData
    calibdata = CalibrationData{typefloat}(roi, cats)
    for (catname, filenames) in [  ("BACK", map(p->p[2], backs_to_download)),
                                   ("FLAT", map(p->p[2], flats_to_download))  ]

        for filename in filenames
            filename = chop(filename; tail=2) * ".gz"
            FitsFile(filename) do fitsfile
                hdu = fitsfile[1]
                realdit = typefloat(hdu["ESO DET SEQ1 REALDIT"].float)
                cube = read(Array{typefloat,3}, hdu, :,:,:)
                sampler = CalibrationFrameSampler(cube, catname, realdit ; roi=roi)
                push!(calibdata, sampler)
            end
        end
    end

    # ReducedCalibration
    firstvalidpixels = findbadpixels(calibdata)
    reduced_calibdata = ReducedCalibration(calibdata ; validpixels=firstvalidpixels)
    findbadpixels!(reduced_calibdata)
    write(goal_reduced_calibdata_filename, reduced_calibdata, overwrite=true)
    run(`gzip -f "$(goal_reduced_calibdata_filename)"`)

    # reduced science
    science_filename = chop(science_to_download[2]; tail=2) * ".gz"
    ppp = PreprocessingParameters(reduced_calibdata; flat="flat", bg="back", Δt=science_dit)
    science_matrix = readfits(Matrix{typefloat}, science_filename, :,:,1)
    (weights, reduced_data) = process(ppp, science_matrix)
    writefits!(goal_reduced_science_filename, FitsHeader(), [ reduced_data ;;; weights ],overwrite=true)
    run(`gzip -f "$(goal_reduced_science_filename)"`)
end
