import numpy as np
import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import optim

from rulframework.model.ABCModel import ABCModel


class BnnModel(ABCModel):

    def __init__(self, model) -> None:
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).double()
        self.optimizer = optim.Adam(self.model.parameters(), lr=.1)
        self.train_losses = None

    def train(self, train_data_x: DataFrame, train_data_y: DataFrame, num_epochs: int = 1000):
        x = torch.tensor(train_data_x.values, dtype=torch.float64).to(self.device)
        y = torch.tensor(train_data_y.values, dtype=torch.float64).to(self.device)
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
                print('epoch: {}/{}'.format(epoch + 1, num_epochs))
                print('Loss: %.4f' % loss.item())
        print('Finished Training')

    def predict(self, input_data: list) -> list:
        input_data = torch.tensor(input_data, dtype=torch.float64).to(self.device)
        with torch.no_grad():
            output = self.model(input_data).tolist()
        return output

    def plot_loss(self):
        plt.plot(range(0, len(self.train_losses)), self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()
