import numpy as np
import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from torch import optim, nn

from rulframework.model.ABCModel import ABCModel


class BnnModel(ABCModel):

    def __init__(self, model: nn.Module) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device=self.device, dtype=torch.float64)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.train_losses = None

    def __call__(self, x: list) -> list:
        pass

    @property
    def loss(self):
        return self.train_losses

    def train(self, train_data_x: ndarray, train_data_y: ndarray, num_epochs: int = 1000):
        x = torch.tensor(train_data_x, dtype=torch.float64, device=self.device)
        y = torch.tensor(train_data_y, dtype=torch.float64, device=self.device)
        hist_epochs = np.zeros((int(num_epochs / 10), 1))
        self.train_losses = np.zeros((int(num_epochs / 10), 1))
        for epoch in range(num_epochs):  # loop over the dataset multiple times
            self.optimizer.zero_grad()
            # forward + backward + optimize
            loss = self.model.sample_elbo(x, y, 1)
            loss.backward()
            self.optimizer.step()
            if epoch % 10 == 0:
                self.train_losses[int(epoch / 10)] = loss.data.cpu()
                hist_epochs[int(epoch / 10)] = epoch + 1
                print('epoch: {}/{}'.format(epoch + 1, num_epochs), end='  ')
                print('Loss: %.4f' % loss.item(), end='\r')
        print('Finished Training')

    def predict(self, input_data: list) -> list:
        x = torch.tensor(input_data, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(x).tolist()
        return output
