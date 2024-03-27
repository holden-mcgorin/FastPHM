from rulframework.entity.Bearing import Bearing, BearingStage
from rulframework.data.stage.eol.ABCEoLCalculator import ABCEoLCalculator
from rulframework.data.stage.fpt.ABCFPTCalculator import ABCFPTCalculator


class BearingStageCalculator:

    def __init__(self, fpt_calculator: ABCFPTCalculator, eol_calculator: ABCEoLCalculator, scale: int) -> None:
        self.fpt_calculator = fpt_calculator
        self.eol_calculator = eol_calculator
        self.scale = scale

    def calculate_state(self, bearing: Bearing):
        fpt_raw, fpt_feature = self.fpt_calculator.get_fpt(bearing.raw_data, bearing.feature_data, self.scale)

        eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature = self.eol_calculator.get_eol(
            bearing.raw_data, bearing.feature_data, self.scale, fpt_raw, fpt_feature)

        bearing.stage_data = BearingStage(fpt_raw, fpt_feature, eol_raw, eol_feature, failure_threshold_raw,
                                          failure_threshold_feature)
