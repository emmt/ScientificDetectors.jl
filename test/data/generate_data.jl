using Downloads, EasyFITS, ScientificDetectors, Random

# DPIDs

DATA_DIR = @__DIR__

dpids_flats = [ "SPHER.2018-05-06T12:01:47.854",
                "SPHER.2018-05-06T12:07:09.316",
                "SPHER.2018-05-06T12:13:13.151"]

dpids_backs = [ "SPHER.2018-05-06T10:26:37.046",
                "SPHER.2018-05-06T11:16:20.263",
                "SPHER.2018-05-06T10:29:55.811"]

dpid_science = "SPHER.2018-05-05T22:58:34.525"


# Paths (.gz will be added during compression)

flats_paths = [ joinpath(DATA_DIR, "flat_1s_db_h23.fits"),
                joinpath(DATA_DIR, "flat_3s_db_h23.fits"),
                joinpath(DATA_DIR, "flat_5s_db_h23.fits")]

backs_paths = [ joinpath(DATA_DIR, "back_1s_db_h23.fits"),
                joinpath(DATA_DIR, "back_8s_db_h23.fits"),
                joinpath(DATA_DIR, "back_96s_db_h23.fits")]

science_path = joinpath(DATA_DIR, "science_96s_db_h23.fits")

reduced_calibdata_path = joinpath(DATA_DIR, "goal_reduced_calib.fits")

reduced_science_path = joinpath(DATA_DIR, "goal_reduced_science_96s_db_h23.fits")


# Download and crop and compress

const ESO_DOWNLOAD_URL = "https://dataportal.eso.org/dataportal_new/file/"

first_x = 480
first_y = 482
first_z = 1
width = 49
height = 45
depth = 10

range_x = first_x : (first_x + width - 1)
range_y = first_y : (first_y + height - 1)
range_z = first_z : (first_z + depth - 1)

mktempdir() do tmpdir
tmpfile = joinpath(tmpdir, "file.fits.Z")

for i in 1:3
    Downloads.download(ESO_DOWNLOAD_URL * dpids_flats[i], tmpfile)
    writefits!(
        flats_paths[i],
        merge(read(FitsHeader, tmpfile), "NAXIS1" => width, "NAXIS2" => height, "NAXIS3" => depth),
        readfits(tmpfile, range_x, range_y, range_z))
    run(`gzip -f $(flats_paths[i])`)
end

for i in 1:3
    Downloads.download(ESO_DOWNLOAD_URL * dpids_backs[i], tmpfile)
    writefits!(
        backs_paths[i],
        merge(read(FitsHeader, tmpfile), "NAXIS1" => width, "NAXIS2" => height, "NAXIS3" => depth),
        readfits(tmpfile, range_x, range_y, range_z))
    run(`gzip -f $(backs_paths[i])`)
end

Downloads.download(ESO_DOWNLOAD_URL * dpid_science, tmpfile)
writefits!(
    science_path,
    merge(read(FitsHeader, tmpfile), "NAXIS" => 2, "NAXIS1" => width, "NAXIS2" => height),
    readfits(tmpfile, range_x, range_y, 1))
run(`gzip -f $science_path`)

end


# basic info
typefloat = FitsFile(f -> f[1].data_eltype, science_path)
roi = DetectorAxes((width, height))
nbpixels = prod(size(roi))
cats = [ CalibrationCategory("BACK", :back), CalibrationCategory("FLAT", :(back + flat)) ]
science_dit = FitsFile(f -> f[1]["ESO DET SEQ1 REALDIT"].float, science_path)

# CalibrationData
Random.seed!(1234)
calibdata = CalibrationData{typefloat}(roi, cats)
for (catname, filepaths) in [ ("BACK", backs_paths), ("FLAT", flats_paths) ]
    for filepath in filepaths
        FitsFile(filepath) do fitsfile
            hdu = fitsfile[1]
            realdit = typefloat(hdu["ESO DET SEQ1 REALDIT"].float)
            cube = read(hdu, (:,:,:))
            sampler = CalibrationFrameSampler(cube, catname, realdit; roi=roi)
            push!(calibdata, sampler)
        end
    end
end

# ReducedCalibration
Random.seed!(1234)
firstvalidpixels = findbadpixels(calibdata)
reduced_calibdata = ReducedCalibration(calibdata;validpixels=firstvalidpixels)
findbadpixels!(reduced_calibdata)

# PreprocessingParameters
Random.seed!(1234)
ppp = PreprocessingParameters(reduced_calibdata; flat="flat", bg="back", Δt=science_dit)
            
# process science
Random.seed!(1234)
data = readfits(Array{typefloat}, science_path)
reduced_data = similar(data)
weights      = similar(data)
for f in size(data,3)
    (w, rd) = process(ppp, view(data,:,:,f))
    reduced_data[:,:,f] .= rd
    weights[:,:,f] .= w
end

# write to disk
write!(reduced_calibdata_path, reduced_calibdata)
writefits!(reduced_science_path, read(FitsHeader, science_path), reduced_data,
                                           FitsHeader("EXTNAME" => "weights"), weights)
run(`gzip -f $reduced_calibdata_path`) # this will add ".gz" to the paths
run(`gzip -f $reduced_science_path`)

