import torch
from numpy import ndarray
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from rulframework.data.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.system.Logger import Logger


class PytorchModel(ABCModel):
    """
    剩余寿命预测模型
    对pytorch神经网络的封装
    """

    @property
    def loss(self) -> list:
        return self.train_losses

    def __init__(self, model: nn.Module, device=None, dtype=None) -> None:
        """
        :param model:pytorch模型
        :param device: 设备（cpu或cuda）
        :param dtype: 参数类型
        """
        # 初始化设备
        if device is None:
            # self.device = torch.device('cpu')
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # 初始化模型参数类型
        if dtype is None:
            self.dtype = torch.float32
        else:
            self.dtype = dtype

        # 初始化模型
        self.model = model.to(device=self.device, dtype=self.dtype)

        # 用于保存每次epoch的训练损失
        self.train_losses = []
        Logger.info(f'\n<< Successfully initialized model:\n\tdevice: {self.device}\n\tdtype: {self.dtype}')

    def __call__(self, x: ndarray) -> ndarray:
        input_data = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.model(input_data)
        return output.cpu().numpy()

    def train(self, train_set: Dataset, epochs=100,
              batch_size=128, weight_decay=0, lr=0.001,
              criterion=None, optimizer=None):
        """
        训练模型
        :param lr:
        :param train_set:
        :param optimizer: 优化器（默认：Adam，学习率0.001）
        :param weight_decay: 正则化系数
        :param batch_size: 批量大小
        :param epochs: 迭代次数
        :param criterion:
        :return:无返回值
        """
        Logger.info('Start training model...')
        # 初始化损失函数
        if criterion is None:
            criterion = nn.MSELoss()

        # 初始化优化器
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)  # 添加正则化项

        x = torch.tensor(train_set.x, dtype=self.dtype, device=self.device)
        y = torch.tensor(train_set.y, dtype=self.dtype, device=self.device)
        if isinstance(criterion, nn.CrossEntropyLoss):
            y = y.squeeze().to(dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

        for epoch in range(epochs):
            self.model.train()  # 设置模型为训练模式
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # 梯度清零
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)  # 保存训练损失

            Logger.debug(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.10f}")
        Logger.info('Model training completed!!!')
