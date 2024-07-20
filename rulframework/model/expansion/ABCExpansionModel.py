from abc import ABC

from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel


class ABCExpansionModel(ABCModel, ABC):
    def __init__(self, model: ABCModel):
        self.model = model

    @property
    def loss(self) -> list:
        return self.model.loss

    def train(self, train_set: Dataset, epochs=100, batch_size=128, weight_decay=0, criterion=None, optimizer=None):
        self.model.train(train_set, epochs, batch_size, weight_decay, criterion, optimizer)
