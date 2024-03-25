from pandas import DataFrame

from rulframework.stage.eol.ABCEoLCalculator import ABCEoLCalculator


class TenMaxAmplitudeEoLCalculator(ABCEoLCalculator):
    def get_eol(self, raw_data: DataFrame, feature_data: DataFrame, scale, fpt_raw, fpt_feature):
        # todo 这里只取第一列做计算fpt，多列情况不适应
        raw_data = raw_data.iloc[:, 0]
        feature_data = feature_data.iloc[:, 0]
        eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature = 0, 0, 0, 0
        normal_stage_raw = raw_data[:fpt_raw]
        failure_threshold_raw = 10 * normal_stage_raw.abs().max()

        for i in range(fpt_raw, len(raw_data)):
            if abs(raw_data[i]) > failure_threshold_raw:
                eol_raw = i
                break
        eol_feature = eol_raw // scale
        failure_threshold_feature = feature_data[eol_feature]
        return eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature
