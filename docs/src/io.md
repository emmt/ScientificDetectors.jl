## Saving calibration or preprocessing parameters

If object `obj` is an instance of `ReducedCalibration` or `PreprocessingParameters`
its contents can be saved into a FITS file by calling:

```julia
write!(path, obj)
```

with `path` the name of the file.  If the FITS file already exists, it is
(silently) overwritten.  Call `write` instead to throw an error if the file
already exists.  The `write` method can take a FITS handle instead of a file
name, to append a new FITS HDU with detector reduced calibration or
pre-processing parameters.


## Loading calibration or preprocessing parameters

Detector reduced calibration data can be loaded from a FITS file by
calling:

```julia
read(ReducedCalibration, src)
```

with `src` the source where to read the data (a file name or a FITS handle).

Similarly pre-processing parameters can be loaded from a FITS file by calling:

```julia
read(PreprocessingParameters, src)
```

with `src` the source where to read the parameters (a file name or a FITS
handle).

The floating-point type, say `T`, and the dimensionality, say `N`, may be
specified to enforce them.  For instance:

```julia
read(ReducedCalibration{Float32,2}, src)
read(PreprocessingParameters{Float64}, src)
```
