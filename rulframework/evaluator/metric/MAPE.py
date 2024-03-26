from rulframework.entity.Bearing import Bearing
from rulframework.evaluator.metric.ABCMetric import ABCMetric


class MAPE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MAPE'

    def measure(self, bearing: Bearing) -> str:
        predicted_values = bearing.predict_history.prediction
        actual_values = bearing.feature_data.iloc[:, 0].tolist()[bearing.stage_data.fpt_feature:bearing.stage_data.eol_feature]
        predicted_values, actual_values = self.trim_lists(predicted_values, actual_values)

        n = len(actual_values)
        total_percentage_error = 0
        for i in range(n):
            if actual_values[i] != 0:
                total_percentage_error += abs((actual_values[i] - predicted_values[i]) / actual_values[i])
            else:
                total_percentage_error += 0

        mape = (total_percentage_error / n) * 100
        return f"{mape:.1f}%"

    @staticmethod
    def trim_lists(list1, list2):
        if len(list1) < len(list2):
            list2 = list2[:len(list1)]
        elif len(list1) > len(list2):
            list1 = list1[:len(list2)]
        return list1, list2