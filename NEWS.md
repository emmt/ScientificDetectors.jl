# User visible changes in `ScientificDetectors` package

## Unreleased

### Breaking changes

- `AstroFITS` is no longer a direct dependency. Reading/writing calibration data,
  pre-processing parameters, sample statistics, etc., from/to FITS file with `readfits`,
  `writefits`, or `writefits!` is now an extension automatically loaded when the
  `AstroFITS` package is loaded.

- Reading/writing calibration data, pre-processing parameters, sample statistics, etc.,
  from/to a FITS file with `Base.read` and `Base.write` has been removed (such a use of
  `write` was previously deprecated). It is now needed to load `AstroFITS` and call
  `readfits`, `writefits`, or `writefits!`.

### Non-breaking changes


## Version v0.3.7

### Changes

- fix ROI writing
- `EasyFITS` replaced by `AstroFITS`.
- `MultivariateOnlineStatistics` replaced by `OnlineSampleStatistics`.

## Version v0.3.6

## Version v0.3.5

- Update compatibility for `EasyFITS`.

## Version v0.3.4

- Update compatibility for `TypeUtils`.

## Version v0.3.3

## Version v0.3.2

## Version v0.3.1

## Version v0.3.0

- Introduce source terms.  The enlightenment in each calibration is a linear
  combination of these terms.

- Use *sufficient statistics* to represent multiple calibration data frames
  acquired under the same conditions (same enlightenment and same exposure
  time).  As a result, memory occupation can be much reduced and reduction of
  calibration data much faster.

- Activate multi-threading for fitting detector parameters when calling
  `ReducedCalibration` constructor on a `CalibrationData` instance.

- Use `TypeUtils` package.

- Rename "good/bad pixels map" as "valid pixels map" or `vpm`: now `1` means a good pixel.
  Improve code around it (type parameter, default value, constructors).

## Version 0.2.2

- Methods `regionofinterest` and `numberofsamples` deprecated in favor of
  `DetectorAxes` and `nobs` (from `StatsBase`).  The deprecated methods are not
  exported.

- Rename pre-processing parameters `p` and `q` (too confusing) as
  `q` and `r` respectively.
