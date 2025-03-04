from rulframework.entity.ABCEntity import Stage
from rulframework.entity.Bearing import Bearing
from rulframework.data.stage.eol.ABCEoLCalculator import ABCEoLCalculator
from rulframework.data.stage.fpt.ABCFPTCalculator import ABCFPTCalculator


class BearingStageCalculator:

    def __init__(self, scale: int, fpt_calculator: ABCFPTCalculator, eol_calculator: ABCEoLCalculator = None) -> None:
        self.fpt_calculator = fpt_calculator
        self.eol_calculator = eol_calculator
        self.scale = scale

    def __call__(self, bearing: Bearing):
        fpt_raw, fpt_feature = self.fpt_calculator(bearing.raw_data, bearing.feature_data, self.scale)

        if self.eol_calculator is None:
            eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature = None, None, None, None
        else:
            eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature = self.eol_calculator(
                bearing.raw_data, bearing.feature_data, self.scale, fpt_raw, fpt_feature)

        bearing.stage_data = Stage(fpt_raw, fpt_feature, eol_raw, eol_feature, failure_threshold_raw,
                                   failure_threshold_feature)
