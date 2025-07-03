from typing import Union, List

from torch.utils.tensorboard import SummaryWriter

from fastphm.model.pytorch.callback.ABCTrainCallback import ABCTrainCallback
import time


class TensorBoardCallback(ABCTrainCallback):
    def __init__(self, log_dir=None):
        if log_dir is None:
            log_dir = f"runs/exp_{time.strftime('%Y%m%d-%H%M%S')}"
        self.writer = SummaryWriter(log_dir=log_dir)

    def on_epoch_end(self, model, epoch: int, losses: Union[float, List[float]]) -> bool:

        # # 记录损失
        # self.writer.add_scalar('Loss/train', losses, epoch)
        # todo 这里只兼容pytorch模型
        for name, param in model.named_parameters():
            # 记录参数
            self.writer.add_histogram(f'Params/{name}', param, epoch)
            self.writer.add_scalar(f'ParamMean/{name}', param.data.mean(), epoch)
            self.writer.add_scalar(f'ParamStd/{name}', param.data.std(), epoch)
            # 记录梯度
            if param.grad is not None:
                self.writer.add_histogram(f'Grads/{name}', param.grad, epoch)
                self.writer.add_scalar(f'GradMean/{name}', param.grad.mean(), epoch)
                self.writer.add_scalar(f'GradStd/{name}', param.grad.std(), epoch)

        return True
