import torch
from torch import nn

from rulframework.model.pytorch.basic.CnnBackbone import CnnBackbone
from rulframework.model.pytorch.basic.FcReluFc import FcReluFc


class Prototype(nn.Module):
    def __init__(self):
        super(Prototype, self).__init__()
        self.backbone = CnnBackbone()
        self.fault_block = FcReluFc([768, 256, 5])
        # self.fault_block = FcReluFcSoftmax([864, 256, 5])
        self.rul_block = FcReluFc([773, 256, 1])

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(-1, 768)
        fault_label = self.fault_block(x)
        rul_label = self.rul_block(torch.cat((x, fault_label), dim=1))
        rul_label = torch.sigmoid(rul_label)
        return fault_label, rul_label
