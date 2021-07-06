# User visible changes in ScientificDetectors

- Methods `regionofinterest` and `numberofsamples` deprecated in favor of
  `DetectorAxes` and `nobs` (from `StatsBase`).  The deprecated methods are not
  exported.

- Rename preprocessing parameters `p` and `q` (too confusing) as
  `q` and `r` respectively.
