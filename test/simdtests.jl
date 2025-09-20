error("not meant to be included. copy paste steps by hand in the REPL.")


# ==============================================================
# calib file list, include this all the time
dpid_flats = [
"SPHER.2022-05-08T09:44:44.628",
"SPHER.2022-05-08T09:45:16.753",
"SPHER.2022-05-08T09:50:16.518"]
dpid_backs = [
"SPHER.2022-05-07T10:39:36.886",
"SPHER.2022-05-07T10:40:57.296",
"SPHER.2022-05-07T10:42:57.778"]


# ==============================================================
# download calib files (do this only once)
using Downloads
isdir("/tmp/toto/") || mkdir("/tmp/toto/")
for dpid in dpid_flats
    Downloads.download("https://dataportal.eso.org/dataportal_new/file/$dpid",
        "/tmp/toto/$dpid.fits.Z")
end
for dpid in dpid_backs
    Downloads.download("https://dataportal.eso.org/dataportal_new/file/$dpid",
        "/tmp/toto/$dpid.fits.Z")
end


# ==============================================================
# computing
using ScientificDetectors, EasyFITS
roi = FitsFile(f -> DetectorAxes(f[1].data_size[1:2]), "/tmp/toto/$(dpid_flats[1]).fits.Z")
cats = [ CalibrationCategory("FLAT", :(flat + back)), CalibrationCategory("BACK", :back) ]
calibdata = CalibrationData{Float64}(roi, cats)
for dpid in dpid_flats
    FitsFile("/tmp/toto/$dpid.fits.Z") do fitsfile
        hdu = fitsfile[1]
        realdit = Float64(hdu["ESO DET SEQ1 REALDIT"].float)
        cube = read(hdu, (:,:,:))
        local sampler
        sampler = CalibrationFrameSampler(cube, "FLAT", realdit; roi=roi)
        push!(calibdata, sampler)
    end
end
for dpid in dpid_backs
    FitsFile("/tmp/toto/$dpid.fits.Z") do fitsfile
        hdu = fitsfile[1]
        realdit = Float64(hdu["ESO DET SEQ1 REALDIT"].float)
        cube = read(hdu, (:,:,:))
        local sampler
        sampler = CalibrationFrameSampler(cube, "BACK", realdit; roi=roi)
        push!(calibdata, sampler)
    end
end
firstvalidpixels = Array{Bool}(findbadpixels(calibdata))
reduced_calibdata = ReducedCalibration(calibdata;validpixels=firstvalidpixels)
findbadpixels!(reduced_calibdata)




# ==============================================================
# writing results to disk
# so we can compare with other results afterwards
using ScientificDetectors, EasyFITS
include("test/calibdataio.jl") # some methods to IO with CalibrationData
write!("/tmp/toto/simd-calibdata.fits", calibdata)
writefits!("/tmp/toto/simd-firstvalidpixels.fits", FitsHeader(), firstvalidpixels)
write!("/tmp/toto/simd-reduced-calibdata.fits", FitsHeader(), reduced_calibdata)




# ==============================================================
# now start again with julia parameter `--check-bounds=yes`
# skip the download part
# skip the writing part too !!!!
# redo the computing part !!!
# then do these tests
using ScientificDetectors, EasyFITS, Test
include("test/calibdataio.jl") # some methods to IO with CalibrationData
simd_calibdata = read(CalibrationData, "/tmp/toto/simd-calibdata.fits")
simd_firstvalidpixels = readfits(Array{Bool}, "/tmp/toto/simd-firstvalidpixels.fits")
simd_reduced_calibdata = read(ReducedCalibration, "/tmp/toto/simd-reduced-calibdata.fits")
@test calibdata == simd_calibdata
@test firstvalidpixels == simd_firstvalidpixels
@test reduced_calibdata == simd_reduced_calibdata







