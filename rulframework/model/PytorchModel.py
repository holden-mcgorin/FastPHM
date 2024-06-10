import torch
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from rulframework.data.dataset.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.predict.Result import Result


class PytorchModel(ABCModel):
    """
    剩余寿命预测模型
    对pytorch神经网络的封装
    """

    @property
    def loss(self) -> list:
        return self.train_losses

    def __init__(self, model: nn.Module, criterion=None) -> None:
        """
        初始化： 模型、评价指标、优化器
        :param model:定义模型结构的类
        :param criterion:默认均方误差
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(device=self.device, dtype=torch.float64)
        # 初始化评价指标
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

        # 用于保存每次epoch的训练损失
        self.train_losses = []

    def train(self, train_data_x: ndarray, train_data_y: ndarray, num_epochs: int = 1000,
              optimizer=None, weight_decay=0):
        """
        训练模型
        :param optimizer: 优化器，默认Adam优化器，学习率0.001
        :param weight_decay: 正则化系数
        :param train_data_x: 训练数据
        :param train_data_y: 训练数据的标签
        :param num_epochs: 迭代次数
        :return:无返回值
        """
        x = torch.tensor(train_data_x, dtype=torch.float64, device=self.device)
        y = torch.tensor(train_data_y, dtype=torch.float64, device=self.device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

        # 初始化优化器
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=weight_decay)  # 添加正则化项
        else:
            optimizer = optimizer

        for epoch in range(num_epochs):
            self.model.train()  # 设置模型为训练模式
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # 梯度清零
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)  # 保存训练损失

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.10f}", end="\r")

    def predict(self, input_data: list) -> list:
        """
        输出一次预测结果
        :param input_data: 输入数据
        :return: 一次预测结果
        """
        input_data = torch.tensor(input_data, dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(input_data).tolist()
        return output

    def end2end_train(self, train_set: Dataset, num_epochs=1000, optimizer=None, weight_decay=0):
        self.train(train_set.x, train_set.y, num_epochs, optimizer, weight_decay)

    def end2end_predict(self, test_set: Dataset) -> Result:
        return Result(mean=self(test_set.x))

    def __call__(self, x: ndarray) -> ndarray:
        input_data = torch.from_numpy(x).to(dtype=torch.float64, device=self.device)
        with torch.no_grad():
            output = self.model(input_data)
        return output.cpu().numpy()
