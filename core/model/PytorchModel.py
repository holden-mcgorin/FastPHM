import torch
from matplotlib import pyplot as plt
from pandas import DataFrame
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from core.predictor.ABCPredictable import ABCPredictable


class PytorchModel(ABCPredictable):
    """
    剩余寿命预测模型
    对pytorch神经网络的封装
    """

    def __init__(self, model: nn.Module, criterion=None, optimizer=None) -> None:
        """
        初始化： 模型、评价指标、优化器
        :param model:定义模型结构的类
        :param criterion:默认均方误差
        :param optimizer:默认Adam优化器，学习率0.001
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device).double()
        # 初始化评价指标
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion
        # 初始化优化器
        if optimizer is None:
            self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        else:
            self.optimizer = optimizer

        # 用于保存每次epoch的训练损失
        self.train_losses = []

    def train(self, train_data_x: DataFrame, train_data_y: DataFrame, num_epochs: int):
        """
        训练模型
        :param train_data_x: 训练数据
        :param train_data_y: 训练数据的标签
        :param num_epochs: 迭代次数
        :return:无返回值
        """
        x = torch.tensor(train_data_x.values, dtype=torch.float64)
        y = torch.tensor(train_data_y.values, dtype=torch.float64)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=32, shuffle=True)

        for epoch in range(num_epochs):
            self.model.train()  # 设置模型为训练模式
            total_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()  # 梯度清零
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                loss.backward()  # 反向传播
                self.optimizer.step()  # 更新权重

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)  # 保存训练损失

            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.10f}")

    def plot_loss(self):
        """
        绘制训练损失曲线
        :return: 无返回值
        """
        plt.plot(range(0, len(self.train_losses)), self.train_losses, label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss Over Epochs')
        plt.legend()
        plt.show()

    def predict(self, input_data: list) -> list:
        """
        输出一次预测结果
        :param input_data: 输入数据
        :return: 一次预测结果
        """
        input_data = torch.tensor(input_data, dtype=torch.float64).to(self.device)
        with torch.no_grad():
            output = self.model(input_data).tolist()
        return output
