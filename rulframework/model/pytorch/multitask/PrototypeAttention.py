import torch
from torch import nn

from rulframework.model.pytorch.basic.CnnBackbone import CnnBackbone
from rulframework.model.pytorch.basic.FcReluFc import FcReluFc


class PrototypeAttention(nn.Module):
    def __init__(self):
        super(PrototypeAttention, self).__init__()
        self.backbone = CnnBackbone()

        # self.project = FcReluFc([864, 512, 128])

        self.fault_block = FcReluFc([768, 512, 128])
        self.fault_block2 = FcReluFc([128, 64, 5])

        self.rul_project = FcReluFc([768, 128, 16])
        # self.lstm = nn.LSTM(1, 16, 2, batch_first=True)
        self.rul_block = FcReluFc([16, 8, 1])

        self.w_q = nn.Linear(5, 16, bias=False)
        # self.w_k = nn.Linear(128, 128, bias=False)
        # self.w_v = nn.Linear(128, 128, bias=False)

    def forward(self, x):
        x = self.backbone(x)
        # x = self.project(x)

        fault_label = self.fault_block(x)
        fault_label = self.fault_block2(fault_label)

        x = self.rul_project(x)
        # 应用注意力机制
        attn_weights = self.attention_weights(fault_label, x)
        attn_output = attn_weights * x

        # attn_output = attn_output.view(-1, 16, 1)
        # lstm_out, (h_n, c_n) = self.lstm(attn_output)
        rul_label = self.rul_block(attn_output)

        rul_label = torch.sigmoid(rul_label)
        # rul_label = self.rul_block(torch.cat((attn_output, fault_label), dim=1))

        return fault_label, rul_label, attn_weights

    def attention_weights(self, q, k):
        q = self.w_q(q)
        # k = self.w_k(key)
        # processor = self.w_v(torch.tanh(q + k))
        feature = torch.tanh(q + k)
        return torch.softmax(feature, dim=1)
