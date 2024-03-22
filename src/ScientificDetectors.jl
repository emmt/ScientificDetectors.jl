#
# ScientificDetectors.jl --
#
# Calibration and pre-processing of data from scientific detectors.
#
#-------------------------------------------------------------------------------
#
# This file if part of the ScientificDetector software
# (https://github.com/emmt/ScientificDetector.jl) licensed under the MIT
# license.
#
# Copyright (C) 2018-2019, Éric Thiébaut.
#

module ScientificDetectors

export
    CalibrationCategory,
    CalibrationData,
    CalibrationDataFrame,
    CalibrationFrameSampler,
    DetectorAxes,
    DetectorAxis,
    IndependentIdenticallyDistributedNoise,
    NoiseModel,
    PreprocessingParameters,
    RealisticNoise,
    ReducedCalibration,
    SampleStatistics,
    SimpleCalibration,
    StaticNoise,
    nobs,
    process!,
    process,
    readfits,
    write!,
    writefits!,
    writefits,
    findbadpixels!,
    findbadpixels

import Base: read, write

using Statistics, StatsBase, Distributions
using MultivariateOnlineStatistics
using ArrayTools, StructuredArrays
using EasyFITS
using EasyFITS: hduname, throw_file_already_exists, FitsImageHDU
import EasyFITS: write!, readfits, writefits, writefits!

function process end
function process! end

include("common.jl")

include("statistics.jl")
import .DetectorStatistics: SampleStatistics

include("calibration.jl")
import .Calibration:
    CalibrationCategory,
    CalibrationData,
    CalibrationDataFrame,
    CalibrationFrameSampler,
    ReducedCalibration,
    SimpleCalibration,
    findbadpixels,
    findbadpixels!

include("preprocessing.jl")
import .Preprocessing:
    IndependentIdenticallyDistributedNoise,
    NoiseModel,
    PreprocessingParameters,
    RealisticNoise,
    StaticNoise

include("io.jl")


@deprecate numberofsamples(A::SampleStatistics) nobs(A) false
@deprecate regionofinterest(A) DetectorAxes(A) false

end # module
