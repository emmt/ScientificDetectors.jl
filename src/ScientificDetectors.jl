#
# ScientificDetectors.jl --
#
# Calibration and pre-processing of data from scientific detectors.
#
#-------------------------------------------------------------------------------
#
# This file is part of the ScientificDetector software
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
    CalibrationDataStat,
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
    findbadpixels!,
    findbadpixels

using Statistics, StatsBase, Distributions
using OnlineSampleStatistics
using ArrayTools, StructuredArrays

using Base: @propagate_inbounds

function process end
function process! end

include("types.jl")
include("axes.jl")
include("common.jl")

include("statistics.jl")
import .DetectorStatistics: SampleStatistics

include("calibration.jl")
import .Calibration:
    CalibrationCategory,
    CalibrationData,
    CalibrationDataFrame,
    CalibrationFrameSampler,
    CalibrationDataStat,
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

@deprecate numberofsamples(A::SampleStatistics) nobs(A) false
@deprecate regionofinterest(A) DetectorAxes(A) false

end # module
