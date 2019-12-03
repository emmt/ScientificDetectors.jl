- `T.(obj)` with `T` any floating-point type works as expected but `@btime`
  reveals that it takes some time and that it allocates some bytes (for a
  400×388 detetctor: 570.816 μs, 2 allocations: 112 bytes).
- Use multi-threading.
- Check what is faster between existing methods `copyto!`, `fill!` and `@simd`
  loops.
- Check what is faster between computing `wgt` and `dat` separately or jointly.
- Calibration procedures can be multi-threaded.
- Replace `calibrate` method by a constructor of `ReducedCalibration`.
- Add means to read/write preprocessing parameters from/to a FITS file.
- Provide simple tools to define bad pixels.
- ROI information in `ReducedCalibration` must be `N`-dimensional.
- Perhaps provide additional information (ROI, exposure time, other detetctor
  parameters, etc.) in the form of a hash-table (or equivalent) in
  `ReducedCalibration` and `PreprocessingParameters` with means to
  read/write/check these information from/to FITS header.
