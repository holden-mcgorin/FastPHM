import torch
import torch.nn as nn
import torch.nn.functional as F


class MS_CAM_1D(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(MS_CAM_1D, self).__init__()

        # 多尺度1D卷积核定义
        self.conv1x1 = nn.Conv1d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv3x1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
        self.conv5x1 = nn.Conv1d(in_channels, in_channels, kernel_size=5, stride=1, padding=2)

        # SE模块：通道注意力部分
        self.fc1 = nn.Linear(in_channels, in_channels // reduction, bias=False)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 多尺度卷积
        x1 = self.conv1x1(x)
        x2 = self.conv3x1(x)
        x3 = self.conv5x1(x)

        # 将不同尺度的卷积结果进行融合
        multi_scale_features = x1 + x2 + x3

        # 通道注意力
        b, c, _ = multi_scale_features.size()
        avg_pool = F.adaptive_avg_pool1d(multi_scale_features, 1).view(b, c)
        channel_attn = self.fc2(F.relu(self.fc1(avg_pool)))
        channel_attn = self.sigmoid(channel_attn).view(b, c, 1)

        # 加权后的输出
        out = multi_scale_features * channel_attn
        return out
