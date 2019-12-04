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
    CalibrationData,
    DetectorAxis,
    IndependentIdenticallyDistributedNoise,
    NoiseModel,
    PreprocessingParameters,
    RealisticNoise,
    ReducedCalibration,
    StaticNoise,
    calibrate,
    process!,
    process,
    write!

import Base: read, write

using Printf

using ArrayTools
using EasyFITS
using EasyFITS: exists, throw_file_already_exists
import EasyFITS: write!, hduname

function calibrate end
function process end
function process! end

include("common.jl")

include("calibration.jl")
import .Calibration: ReducedCalibration, CalibrationData

include("preprocessing.jl")
import .Preprocessing:
    IndependentIdenticallyDistributedNoise,
    NoiseModel,
    PreprocessingParameters,
    RealisticNoise,
    StaticNoise

include("io.jl")

end # module
