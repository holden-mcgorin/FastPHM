import torch
import torch.nn as nn

class DNNWithDropout(nn.Module):
    def __init__(self):
        super(DNNWithDropout, self).__init__()
        self.fc1 = nn.Linear(7, 5)  # 输入层到隐藏层
        self.dropout = nn.Dropout(0.5)   # Dropout 层，丢弃概率为 0.5
        self.fc2 = nn.Linear(5, 3)  # 隐藏层到输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 输入层到隐藏层，使用 ReLU 激活函数
        x = self.dropout(x)          # Dropout 层
        x = torch.relu(self.fc2(x))  # 隐藏层到输出层，使用 ReLU 激活函数
        return x

# 创建模型实例
model = DNNWithDropout()

# 将模型设置为评估模式（测试模式）
model.train()

# 获取模型输出，保持 Dropout 层
with torch.no_grad():  # 禁用梯度计算
    input_data = torch.randn(1, 7)  # 示例输入数据
    output = model(input_data)

print("模型输出:", output)
