# Julia package for working with scientific detectors

`ScientificDetectors` is a [Julia][julia-url] package for calibrating and
pre-processing data from scientific detectors.

## Usage

### Reduce calibration data

Assuming `dir` is the path to the directory where are stored FITS files with
calibration data, reduced calibration data can be obtained as follows:

```julia
using ScientificDetectors, AstronomicalDetectors, Glob
list = scan_calibrations(glob("SPHER.2015-12-2*", dir))
data = read(CalibrationData{Float64}, list; part=(501:580,601:650))
calib = ReducedCalibration(data)
```

Then reduced calibration data can be saved into a FITS file for further use:

```julia
write("calib.fits", calib)
```

In order to read the reduced calibration data, call:

```julia
calib = read(ReducedCalibration, "calib.fits.gz");
```

### Using reduced calibration

To read the reduced calibration data:

```julia
using ScientificDetectors
calib = read(ReducedCalibration, "calib.fits.gz")
```

Convert reduced calibration into pre-processing parameters for 3 seconds
exposure time:

```julia
prm = PreprocessingParameters(calib; flat="FLAT_FIELD_RAW_2",
                              bg="DARK_RAW", Δt=3);
```

Idem but explicitly specifying the floating-point type:

```julia
prm = PreprocessingParameters{Float32}(calib; flat="FLAT_FIELD_RAW_2",
                                       bg="DARK_RAW", Δt=3);
```

Note that a second argument (after `calib`) may be specified to indicate the
location of defective pixels.  This optional argument is an array of booleans
of same size as the data frames and with false values where pixels should be
considered as defective.  In addition to this valid pixels map, pixels for which
the calibration parameters yield invalid pre-processing parameters are also
considered as defective.

Read some raw data and apply pre-processings to the 7-th raw frame:

```julia
using AstroFITS
raw = read(FitsImage, "SPHER.2018-06-22T04.00.04.544IRD_SCIENCE_DPI_RAW.fits.gz");
wgt, dat = process(prm, raw[:,:,7]);
```

this yields 2 arrays: `dat` contains the pre-processed pixel values while `wgt`
gives their statistical weights.  To avoid allocating arrays, the `process!`
method can be called:

```julia
process!(wgt, dat, prm, raw[:,:,7]);
```

## Structures Hierarchy

- `SampleStatistics` delete?

- create `CalibrationData` with ROI and calibration categories with source to categories
  matrix;
- entry point: collect. `CalibrationDataFrame` for different Δt and categories
  in `CalibrationData`;
- fit detector parameters -> `ReducedCalibration`

## Installation

The easiest way to install `ScientificDetectors` is via Julia registry
[`EmmtRegistry`](https://github.com/emmt/EmmtRegistry):

```julia
using Pkg
pkg"registry add https://github.com/emmt/EmmtRegistry"
pkg"add ScientificDetectors"
```

[doc-stable-img]: https://img.shields.io/badge/docs-stable-blue.svg
[doc-stable-url]: https://emmt.github.io/ScientificDetectors.jl/stable

[doc-dev-img]: https://img.shields.io/badge/docs-dev-blue.svg
[doc-dev-url]: https://emmt.github.io/ScientificDetectors.jl/dev

[license-url]: ./LICENSE.md
[license-img]: http://img.shields.io/badge/license-MIT-brightgreen.svg?style=flat

[travis-img]: https://travis-ci.org/emmt/ScientificDetectors.jl.svg?branch=master
[travis-url]: https://travis-ci.org/emmt/ScientificDetectors.jl

[appveyor-img]: https://ci.appveyor.com/api/projects/status/github/emmt/ScientificDetectors.jl?branch=master
[appveyor-url]: https://ci.appveyor.com/project/emmt/ScientificDetectors-jl/branch/master

[coveralls-img]: https://coveralls.io/repos/emmt/ScientificDetectors.jl/badge.svg?branch=master&service=github
[coveralls-url]: https://coveralls.io/github/emmt/ScientificDetectors.jl?branch=master

[codecov-img]: http://codecov.io/github/emmt/ScientificDetectors.jl/coverage.svg?branch=master
[codecov-url]: http://codecov.io/github/emmt/ScientificDetectors.jl?branch=master

[julia-url]: https://julialang.org/
[julia-pkgs-url]: https://pkg.julialang.org/
