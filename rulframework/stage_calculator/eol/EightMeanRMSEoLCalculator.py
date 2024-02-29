from pandas import DataFrame

from rulframework.stage_calculator.eol.ABCEoLCalculator import ABCEoLCalculator


class EightMeanRMSEoLCalculator(ABCEoLCalculator):
    def get_eol(self, raw_data: DataFrame, feature_data: DataFrame, scale, fpt_raw, fpt_feature):
        # todo 这里只取第一列做计算fpt，多列情况不适应
        raw_data = raw_data.iloc[:, 0]
        feature_data = feature_data.iloc[:, 0]
        eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature = 0, 0, 0, 0
        normal_stage_feature = feature_data[:fpt_feature]
        failure_threshold_feature = 8 * normal_stage_feature.abs().mean()
        print(normal_stage_feature.abs().mean())

        for i in range(fpt_feature, len(feature_data)):
            if abs(feature_data[i]) > failure_threshold_feature:
                eol_feature = i
                break
        eol_raw = eol_raw * scale
        failure_threshold_raw = raw_data[eol_raw]
        return eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature
