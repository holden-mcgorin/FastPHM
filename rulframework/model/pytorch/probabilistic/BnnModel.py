import numpy as np
import torch
from numpy import ndarray
from torch import optim, nn

from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel


class BnnModel(ABCModel):

    def __init__(self, model: nn.Module, device=None, dtype=None) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device=self.device, dtype=torch.float64)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_losses = None
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

    def __call__(self, x: list) -> list:
        pass

    @property
    def loss(self):
        return self.train_losses

    def train(self, train_set: Dataset, epochs=100,
              batch_size=128, weight_decay=0,
              criterion=None, optimizer=None):
        x = torch.tensor(train_set.x, dtype=self.dtype, device=self.device)
        y = torch.tensor(train_set.y, dtype=self.dtype, device=self.device)

        hist_epochs = np.zeros((int(epochs / 10), 1))
        self.train_losses = np.zeros((int(epochs / 10), 1))
        for epoch in range(epochs):  # loop over the label multiple times
            self.optimizer.zero_grad()
            # forward + backward + optimize
            loss = self.model.sample_elbo(x, y, 1)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                self.train_losses[int(epoch / 10)] = loss.data.cpu()
                hist_epochs[int(epoch / 10)] = epoch + 1
                print('epoch: {}/{}'.format(epoch + 1, epochs), end='  ')
                print('Loss: %.4f' % loss.item(), end='\r')
        print('Finished Training')

    def predict(self, input_data: list) -> list:
        x = torch.tensor(input_data, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(x).tolist()
        return output
