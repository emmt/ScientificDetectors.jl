error("not meant to be included. copy paste steps by hand in the REPL.")

# this script it supposed to be executed twice,

# once with:
# julia --project --threads=1 --check-bounds=auto

# and once with:
# julia --project --threads=1 --check-bounds=yes

# some parts of the script are with check-bounds=auto, and others with check-bounds=yes
# you are supposed to compare the two sets of produced data then



# ==============================================================
# [both cases] calib file list
dpid_flats = [
"SPHER.2022-05-08T09:44:44.628",
"SPHER.2022-05-08T09:45:16.753",
"SPHER.2022-05-08T09:50:16.518"]
dpid_backs = [
"SPHER.2022-05-07T10:39:36.886",
"SPHER.2022-05-07T10:40:57.296",
"SPHER.2022-05-07T10:42:57.778"]



# ==============================================================
# [both cases, do it only once] download calib files
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
# [both cases] computing
using ScientificDetectors, EasyFITS, Random
Random.seed!(1234)
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
firstvalidpixels = Array{Bool}(findbadpixels(calibdata));
#                                                            WE COPY !!! IMPORTANT !!!
#                                                  because we will reuse `firstvalidpixels`
reduced_calibdata = ReducedCalibration(calibdata;validpixels=copy(firstvalidpixels))
findbadpixels!(reduced_calibdata)




# ==============================================================
# [case check-bounds=auto] writing results to disk
# so we can compare with other results afterwards
using ScientificDetectors, EasyFITS
include("test/calibdataio.jl") # some methods to IO with CalibrationData
write!("/tmp/toto/simd-calibdata.fits", calibdata)
writefits!("/tmp/toto/simd-firstvalidpixels.fits", FitsHeader(), firstvalidpixels)
write!("/tmp/toto/simd-reduced-calibdata.fits", FitsHeader(), reduced_calibdata)

# ==============================================================
# [case check-bounds=yes] writing results to disk
# so we can compare with other results afterwards
using ScientificDetectors, EasyFITS
include("test/calibdataio.jl") # some methods to IO with CalibrationData
write!("/tmp/toto/seq-calibdata.fits", calibdata)
writefits!("/tmp/toto/seq-firstvalidpixels.fits", FitsHeader(), firstvalidpixels)
write!("/tmp/toto/seq-reduced-calibdata.fits", FitsHeader(), reduced_calibdata)




# ==============================================================
# [case comparison of results]
using ScientificDetectors, EasyFITS, Test
include("test/calibdataio.jl") # some methods to IO with CalibrationData
simd_calibdata = read(CalibrationData, "/tmp/toto/simd-calibdata.fits");
simd_firstvalidpixels = readfits(Array{Bool}, "/tmp/toto/simd-firstvalidpixels.fits");
simd_reduced_calibdata = read(ReducedCalibration, "/tmp/toto/simd-reduced-calibdata.fits");
seq_calibdata = read(CalibrationData, "/tmp/toto/seq-calibdata.fits");
seq_firstvalidpixels = readfits(Array{Bool}, "/tmp/toto/seq-firstvalidpixels.fits");
seq_reduced_calibdata = read(ReducedCalibration, "/tmp/toto/seq-reduced-calibdata.fits");
@test seq_calibdata == simd_calibdata
@test seq_firstvalidpixels == simd_firstvalidpixels
@test seq_reduced_calibdata == simd_reduced_calibdata
@test count(.!(.≈(seq_reduced_calibdata.f, simd_reduced_calibdata.f; rtol=0.01, atol=10))) == 0
@test count(.!(.≈(seq_reduced_calibdata.g, simd_reduced_calibdata.g; rtol=0.01, atol=0.001))) == 0
@test count(.!(.≈(seq_reduced_calibdata.z, simd_reduced_calibdata.z; rtol=0.01, atol=0.01))) == 0
@test count(.!(.≈(seq_reduced_calibdata.σ, simd_reduced_calibdata.σ; rtol=0.01, atol=0.01))) == 0
@test count(.!(.≈(seq_reduced_calibdata.s[1], simd_reduced_calibdata.s[1]; rtol=0.01, atol=0.001))) == 0
@test count(.!(.≈(seq_reduced_calibdata.s[2], simd_reduced_calibdata.s[2]; rtol=0.01, atol=0.001))) == 0
@test count(.!(.==(seq_reduced_calibdata.vpm, simd_reduced_calibdata.vpm))) == 0














# following part was a check on the valid pixels process, but it turns out they are the same
# so i just keep it for archive, no need to execute it

# ==============================================================
# part to check findvalidpixels intermediary values
# start julia with no parameter
using StatsBase, Distributions, Random
Random.seed!(1234)
using LinearAlgebra: Symmetric, cholesky
nbsrc = length(calibdata.stat)
numpix = prod(size(calibdata))
D = zeros(numpix, nbsrc);
@inbounds for i in 1:nbsrc
    D[:,i] = mean(calibdata.stat[i])[:]
end
meanbysrc = mean(D, dims=1)
E = (D .- meanbysrc);
F = Symmetric(1/(numpix-1) * (E'*E))
G = cholesky(F; check=false)
X2 = sum(abs2, E / G.U; dims=2);
VPM = (X2 .< cquantile(Chisq(nbsrc), 0.05));
Array{Bool}(reshape(VPM, 2048, 1024)) == firstvalidpixels || error("!!!! DIFFERENT CODES !!!!!")
# one particular pixel who is different between the two methods
x = 810; y = 521; i = x + ((y-1)*2048)
Dpix = D[i,:]
X2pix = X2[i]


# ==============================================================
# writing results to disk
# so we can compare with other results afterwards
write("/tmp/toto/smid-meanbysrc.jl", string(meanbysrc))
write("/tmp/toto/smid-F.jl", string(F))
write("/tmp/toto/smid-G.jl", string(G))
writefits!("/tmp/toto/smid-X2.fits", FitsHeader(), reshape(X2, 2048, 1024))
write("/tmp/toto/smid-Dpix.jl", string(Dpix))
write("/tmp/toto/smid-X2pix.jl", string(X2pix))


# ==============================================================
# here restart with julia parameter `--check-bounds=yes` and skip write to disk part
# now verify these tests
using Test, LinearAlgebra
include("/tmp/toto/smid-meanbysrc.jl")
smid_meanbysrc = ans;
include("/tmp/toto/smid-F.jl")
smid_F = ans;
include("/tmp/toto/smid-G.jl")
smid_G = ans;
include("/tmp/toto/smid-Dpix.jl")
smid_Dpix = ans;
include("/tmp/toto/smid-X2pix.jl")
smid_X2pix = ans;
@test meanbysrc == smid_meanbysrc
@test smid_F == F
@test Matrix(smid_G) == Matrix(G)
@test smid_Dpix == Dpix
@test smid_X2pix == X2pix
@test meanbysrc ≈ smid_meanbysrc
@test smid_F ≈ F
@test Matrix(smid_G) ≈ Matrix(G)
@test smid_X2pix ≈ X2pix











