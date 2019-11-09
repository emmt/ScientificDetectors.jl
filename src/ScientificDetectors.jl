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
    write!

function calibrate end
function process end
function process! end
function write! end

include("calibration.jl")
import .Calibration: ReducedCalibration

include("preprocessing.jl")
import .Preprocessing: PreprocessingParameters

end # module
