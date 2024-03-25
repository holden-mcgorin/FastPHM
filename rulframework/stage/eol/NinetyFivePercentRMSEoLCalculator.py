from pandas import DataFrame

from rulframework.stage.eol.ABCEoLCalculator import ABCEoLCalculator


class NinetyFivePercentRMSEoLCalculator(ABCEoLCalculator):

    def get_eol(self, raw_data: DataFrame, feature_data: DataFrame, scale: int, fpt_raw: int, fpt_feature: int) -> (
            int, int, int, int):
        # todo 这里只取第一列做计算fpt，多列情况不适应
        raw_data = raw_data.iloc[:, 0]
        feature_data = feature_data.iloc[:, 0]

        eol_feature = round(len(feature_data) * 0.95)
        eol_raw = round(len(raw_data) * 0.95)
        failure_threshold_raw = raw_data.iloc[eol_raw]

        # 二次修正eol，因为有可能eol前已经有数据超过失效阈值，需要将eol提前
        before_eol_feature_data = feature_data[:eol_feature + 1]
        eol_feature = before_eol_feature_data.idxmax()
        failure_threshold_feature = feature_data.iloc[eol_feature]

        return eol_raw, eol_feature, failure_threshold_raw, failure_threshold_feature
