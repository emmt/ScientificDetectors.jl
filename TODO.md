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
- Provide simple tools to defined bad pixels.
