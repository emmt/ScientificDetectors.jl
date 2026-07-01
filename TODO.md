# List of wishes

- In `affinecorrection` where `y = (x - b)*a` divide by the median of `a` to have `y` in
  ADU, optionally, divide by `Δt` to be in ADU/s. cf. line 340 in `preprocessing.jl`.

- Check for which (old) versions of Julia the tests pass.

- Make `AstroFITS` an extension of `ScientificDetectors`.

- Move everything depending on `FITS` and `AstroFITS` to package `AstronomicalDetectors`.
- Calibration procedures can be multi-threaded.
- Add progress bar in `ReducedCalibration`.
- The affine correction performed in the preprocessing writes `(raw[i] - b[i])*a[i]` which
  does not exploit fused-multiply-add (FMA) instructions; rewrite it as `raw[i]*a[i] -
  ab[i]` where `ab[i] = a[i]*b[i]`.
- `T.(obj)` with `T` any floating-point type works as expected but `@btime` reveals that it
  takes some time and that it allocates some bytes (for a 400×388 detector: 570.816 μs, 2
  allocations: 112 bytes).
- Check what is faster between existing methods `copyto!`, `fill!` and `@simd` loops.
- Check what is faster between computing `wgt` and `dat` separately or jointly.
- Replace `calibrate` method by a constructor of `ReducedCalibration`.
- Provide simple tools to define bad pixels.
- ROI information in `ReducedCalibration` must be `N`-dimensional.
- Perhaps provide additional information (ROI, exposure time, other detector parameters,
  etc.) in the form of a hash-table (or equivalent) in `ReducedCalibration` and
  `PreprocessingParameters` with means to read/write/check these information from/to FITS
  header.
- Add fitting of a smooth model for the flat.
