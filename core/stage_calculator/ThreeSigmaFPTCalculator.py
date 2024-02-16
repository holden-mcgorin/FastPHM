import numpy as np


class ThreeSigmaFPTCalculator:
    def get_fpt(self, signal):
        fpt = 0
        for i in range(1, len(signal)):
            sliced_list = signal[:i]
            if max(sliced_list) > self.__mean_plus_3std(sliced_list):
                fpt = i
                break
        return fpt

    def __mean_plus_3std(self, signal):
        mean_value = np.mean(signal)
        std_dev = np.std(signal)
        result = mean_value + 3 * std_dev
        return result
