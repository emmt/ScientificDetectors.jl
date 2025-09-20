error("not meant to be included. copy paste steps by hand in the REPL.")




# ==============================================================
# download calib files (do this only once)
using Downloads
mkdir("/tmp/toto/")
Downloads.download("https://dataportal.eso.org/dataportal_new/file/SPHER.2022-03-30T10:53:39.626",
    "/tmp/toto/f1.fits.Z")
Downloads.download("https://dataportal.eso.org/dataportal_new/file/SPHER.2022-03-30T10:55:00.262",
    "/tmp/toto/f2.fits.Z")
Downloads.download("https://dataportal.eso.org/dataportal_new/file/SPHER.2022-03-30T10:57:01.131",
    "/tmp/toto/f3.fits.Z")
Downloads.download("https://dataportal.eso.org/dataportal_new/file/SPHER.2022-03-30T10:59:41.768",
    "/tmp/toto/f4.fits.Z")
Downloads.download("https://dataportal.eso.org/dataportal_new/file/SPHER.2022-03-30T11:03:02.619",
    "/tmp/toto/f5.fits.Z")




# ==============================================================
# computing
using ScientificDetectors, EasyFITS
roi = FitsFile(f -> DetectorAxes(f[1].data_size[1:2]), "/tmp/toto/f1.fits.Z")
cats = [ CalibrationCategory("FLAT", :flat) ]
calibdata = CalibrationData{Float64}(roi, cats)
for filepath in [ "/tmp/toto/f1.fits.Z", "/tmp/toto/f2.fits.Z", "/tmp/toto/f3.fits.Z",
                   "/tmp/toto/f4.fits.Z", "/tmp/toto/f5.fits.Z" ]
    FitsFile(filepath) do fitsfile
        hdu = fitsfile[1]
        realdit = Float64(hdu["ESO DET SEQ1 REALDIT"].float)
        cube = read(hdu, (:,:,:))
        local sampler
        sampler = CalibrationFrameSampler(cube, "FLAT", realdit; roi=roi)
        push!(calibdata, sampler)
    end
end
firstvalidpixels = findbadpixels(calibdata)
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
# now reload this script with julia parameter `--check-bounds=yes`
# skip the download part
# skip the writing part too !!!!
# redo the computing part !!!
# then do these tests
using ScientificDetectors, EasyFITS, Test
include("test/calibdataio.jl") # some methods to IO with CalibrationData
simd_calibdata = read(CalibrationData, "/tmp/toto/simd-calibdata.fits")
simd_firstvalidpixels = readfits("/tmp/toto/simd-firstvalidpixels.fits")
simd_reduced_calibdata = read(ReducedCalibration, "/tmp/toto/simd-reduced-calibdata.fits")
@test calibdata == simd_calibdata
@test firstvalidpixels == simd_firstvalidpixels
@test reduced_calibdata == simd_reduced_calibdata







