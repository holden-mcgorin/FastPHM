from rulframework.entity.Bearing import Bearing
from rulframework.predict.evaluator.metric.ABCMetric import ABCMetric


class MSE(ABCMetric):
    @property
    def name(self) -> str:
        return 'MSE'

    def measure(self, bearing: Bearing) -> str:
        predicted_values = bearing.predict_history.prediction
        actual_values = bearing.feature_data.iloc[:, 0].tolist()[bearing.stage_data.fpt_feature:bearing.stage_data.eol_feature]
        predicted_values, actual_values = self.trim_lists(predicted_values, actual_values)

        n = len(actual_values)
        squared_errors = [(actual_values[i] - predicted_values[i]) ** 2 for i in range(n)]
        mse = sum(squared_errors) / n
        return f"{mse:.4f}"

    @staticmethod
    def trim_lists(list1, list2):
        if len(list1) < len(list2):
            list2 = list2[:len(list1)]
        elif len(list1) > len(list2):
            list1 = list1[:len(list2)]
        return list1, list2
