from rulframework.data.Dataset import Dataset
from rulframework.entity.Bearing import Bearing
from rulframework.exp.ABCExperiment import ABCExperiment
from rulframework.model.ABCModel import ABCModel
from rulframework.util.Cache import Cache


class CacheExperiment(ABCExperiment):
    def __init__(self, exp: ABCExperiment, prefix: str, cache_dataset: bool = True, cache_model: bool = False):
        self.exp = exp
        self.cache_dataset = cache_dataset
        self.cache_model = cache_model
        self.prefix = prefix

    def get_dataset(self, bearings: [Bearing], base_bearing: Bearing = None) -> (Dataset, Dataset):
        train_set, test_set = None, None
        if self.cache_dataset:
            train_set = Cache.load(self.prefix + '_train_' + str(bearings))
            test_set = Cache.load(self.prefix + '_test_' + str(bearings))

        if train_set is None or test_set is None:
            train_set, test_set = self.exp.get_dataset(bearings, base_bearing)
            Cache.save(train_set, self.prefix + '_train_' + str(bearings))
            Cache.save(test_set, self.prefix + '_test_' + str(bearings))

        return train_set, test_set

    def get_model(self, train_set: Dataset) -> ABCModel:
        model = None
        if self.cache_model:
            model = Cache.load(self.prefix + '_model')
        if model is None:
            model = self.exp.get_model(train_set)

        return model

    def test(self, model, test_set: Dataset) -> float:
        return self.exp.test(model, test_set)
