# User visible changes in `ScientificDetectors` package

## Version v0.3.0

- Introduce source terms.  The enlightenment in each calibration is a linear
  combination of these terms.

- Use *sufficient statistics* to represent multiple calibration data frames
  acquired under the same conditions (same enlightenment and same exposure
  time).  As a result, memory occupation can be much reduced and reduction of
  calibration data much faster.

- Activate multi-threading for fitting detector parameters when calling
  `ReducedCalibration` constructor on a `CalibrationData` instance.

- Use `AsType` package.

## Version 0.2.2

- Methods `regionofinterest` and `numberofsamples` deprecated in favor of
  `DetectorAxes` and `nobs` (from `StatsBase`).  The deprecated methods are not
  exported.

- Rename preprocessing parameters `p` and `q` (too confusing) as
  `q` and `r` respectively.
