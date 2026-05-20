using Test, ScientificDetectors, AstroFITS, YAML, StatsBase

@testset "CalibrationData_from_yaml.jl" begin
mktempdir() do tmpdir

write(normpath(tmpdir, "config.yml"),
"""
roi:
    - length: 2
      offset: 0
    - length: 4
      bin: 1
datahdu: 1
categories:
    FLAT:
        sources: flat + dark
        files:
            1.0:
                - "flat1s-1.fits"
                - "flat1s-2.fits"
            10.0:
                - "flat10s.fits"
    DARK:
        sources: dark
        datahdu: "darkcube"
        files:
            10.0:
                - "dark10s.fits"
            100.0:
                - "dark100s-1.fits"
                - "dark100s-2.fits"
""")

writefits!(normpath(tmpdir, "flat1s-1.fits"), (), 1010 .+ rand(2, 4))
writefits!(normpath(tmpdir, "flat1s-2.fits"), (), 1010 .+ rand(2, 4, 8))
writefits!(normpath(tmpdir, "flat10s.fits"), (), 10100 .+ rand(2, 4, 5))
writefits!(normpath(tmpdir, "dark10s.fits"), (), zeros(1),
    FitsHeader("EXTNAME" => "darkcube"), 10 .+ rand(2, 4, 30))
writefits!(normpath(tmpdir, "dark100s-1.fits"), (), zeros(1),
    FitsHeader("EXTNAME" => "darkcube"), 100 .+ rand(2, 4, 2))
writefits!(normpath(tmpdir, "dark100s-2.fits"), (), zeros(1),
    FitsHeader("EXTNAME" => "darkcube"), 100 .+ rand(2, 4, 1))

calib_data = CalibrationData{Float32}(normpath(tmpdir, "config.yml"); basedir=tmpdir)
@test calib_data isa CalibrationData
@test Set(keys(calib_data.cat_index)) == Set(["FLAT", "DARK"])
@test Set(keys(calib_data.src_index)) == Set(["flat", "dark"])
@test Set(keys(calib_data.stat_index)) == Set([("FLAT",1.0), ("FLAT",10.0),
                                               ("DARK",10.0), ("DARK",100.0)])
@test size(calib_data.stat[calib_data.stat_index[("FLAT",1.0)]]) == (2,4)
@test isapprox(median(mean(calib_data.stat[calib_data.stat_index[("FLAT",1.0)]])), 1010; atol=1)
@test isapprox(median(mean(calib_data.stat[calib_data.stat_index[("FLAT",10.0)]])), 10100; atol=1)
@test isapprox(median(mean(calib_data.stat[calib_data.stat_index[("DARK",10.0)]])), 10; atol=1)
@test isapprox(median(mean(calib_data.stat[calib_data.stat_index[("DARK",100.0)]])), 100; atol=1)
@test nobs(calib_data.stat[calib_data.stat_index[("FLAT",1.0)]])[1] == 9
@test nobs(calib_data.stat[calib_data.stat_index[("FLAT",10.0)]])[1] == 5
@test nobs(calib_data.stat[calib_data.stat_index[("DARK",10.0)]])[1] == 30
@test nobs(calib_data.stat[calib_data.stat_index[("DARK",100.0)]])[1] == 3

end
end
