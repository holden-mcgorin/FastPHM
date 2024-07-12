from typing import Union, List

import torch
from numpy import ndarray
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

from rulframework.data.dataset.Dataset import Dataset
from rulframework.model.ABCModel import ABCModel
from rulframework.predict.Result import Result
from rulframework.system.Logger import Logger


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
        :param criterion: 损失函数（默认：MSE）
        :param model:定义模型结构的类
        :param criterion:默认均方误差
        """
        # self.device = torch.device('cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = torch.float32
        Logger.info(f'\n  当前使用设备：{self.device}\n  模型参数类型：{self.dtype}')
        self.model = model.to(device=self.device, dtype=self.dtype)
        # 用于保存每次epoch的训练损失
        self.train_losses = []

        # 初始化损失函数
        if criterion is None:
            self.criterion = nn.MSELoss()
        else:
            self.criterion = criterion

    def __call__(self, x: ndarray) -> Union[tuple, ndarray]:
        input_data = torch.from_numpy(x).to(dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.model(input_data)

        # 当模型是多输出时，将每个输出的张量转ndarray
        if isinstance(output, tuple):
            return tuple(e.cpu().numpy() for e in output)
        else:
            return output.cpu().numpy()

    def train(self, train_data_x: ndarray, train_data_y: ndarray,
              num_epochs: int = 1000, batch_size=128,
              optimizer=None, weight_decay=0):
        """
        训练模型
        :param optimizer: 优化器（默认：Adam，学习率0.001）
        :param weight_decay: 正则化系数
        :param train_data_x: 训练数据
        :param train_data_y: 训练数据的标签
        :param batch_size:
        :param num_epochs: 迭代次数
        :return:无返回值
        """
        x = torch.tensor(train_data_x, dtype=self.dtype, device=self.device)
        y = torch.tensor(train_data_y, dtype=self.dtype, device=self.device)
        if isinstance(self.criterion, nn.CrossEntropyLoss):
            y = y.squeeze().to(dtype=torch.long)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

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

            Logger.debug(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.10f}")

    def predict(self, input_data: list) -> list:
        """
        输出一次预测结果
        :param input_data: 输入数据
        :return: 一次预测结果
        """
        input_data = torch.tensor(input_data, dtype=self.dtype, device=self.device)
        with torch.no_grad():
            output = self.model(input_data).tolist()
        return output

    def end2end_train(self, train_set: Dataset, num_epochs=1000, batch_size=128, optimizer=None, weight_decay=0):
        self.train(train_set.x, train_set.y, num_epochs, batch_size, optimizer, weight_decay)

    def end2end_predict(self, test_set: Dataset) -> Result:
        return Result(mean=self(test_set.x))

    def multi_label_train(self, train_set: Dataset, epoch: int, criterion: List[nn.Module], optimizer=None):
        x = torch.tensor(train_set.x, dtype=self.dtype, device=self.device)
        y = torch.tensor(train_set.y, dtype=self.dtype, device=self.device)
        train_loader = DataLoader(TensorDataset(x, y), batch_size=64, shuffle=True)

        # 验证合法性
        if len(criterion) != train_set.num_label:
            Logger.critical(f'标签种类：{train_set.num_label} 与损失函数数量：{len(criterion)}不一致')

        # 初始化优化器
        if optimizer is None:
            optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=0)  # 添加正则化项
        else:
            optimizer = optimizer

        for e in range(epoch):
            self.model.train()
            total_loss = 0.0
            for inputs, labels in train_loader:
                optimizer.zero_grad()  # 梯度清零
                outputs = self.model(inputs)
                losses = []
                for i, indices in enumerate(train_set.sub_label_map.values()):
                    if isinstance(criterion[i], nn.CrossEntropyLoss):
                        loss = criterion[i](outputs[i], labels[:, indices[0]:indices[1]].squeeze().to(dtype=torch.long))
                    else:
                        loss = criterion[i](outputs[i], labels[:, indices[0]:indices[1]])
                    losses.append(loss)
                sum_loss = sum(losses)
                sum_loss.backward()  # 反向传播
                optimizer.step()  # 更新权重

                total_loss += sum_loss.item()

            avg_loss = total_loss / len(train_loader)
            self.train_losses.append(avg_loss)  # 保存训练损失

            Logger.debug(f"Epoch {e + 1}/{epoch}, Loss: {avg_loss:.10f}")

    def multi_label_predict(self, test_set: Dataset) -> tuple:
        all_result = self(test_set.x)
        results = []
        for e in all_result:
            results.append(Result(mean=e))
        return tuple(results)
