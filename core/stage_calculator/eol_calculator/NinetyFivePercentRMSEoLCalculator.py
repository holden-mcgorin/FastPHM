from core.stage_calculator.eol_calculator.ABCEoLCalculator import ABCEoLCalculator


class NinetyFivePercentRMSEoLCalculator(ABCEoLCalculator):
    def get_eol(self, raw_data, feature_data, scale, fpt_raw, fpt_feature) -> (int, int, int, int):
        eol_feature = round(len(feature_data) * 0.95)
        eol_raw = round(len(raw_data) * 0.95)
        failure_threshold_feature = feature_data.iloc[eol_feature, 0]
        failure_threshold_raw = raw_data.iloc[eol_raw, 0]
        return eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature
