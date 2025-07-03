from typing import Union, List

from fastphm.model.pytorch.callback.ABCTrainCallback import ABCTrainCallback

from fastphm.system.Logger import Logger


class CheckGradientsCallback(ABCTrainCallback):
    """
    梯度检查回调器
    当梯度很小时日志输出警告
    """

    def __init__(self, min_threshold=1e-5, max_threshold=1e+3):
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def on_epoch_end(self, model, epoch: int, avg_loss: Union[float, List[float]]) -> bool:
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_mean = param.grad.abs().mean().item()
                if grad_mean < self.min_threshold:
                    Logger.warning(f"[CheckGradients]  {name} gradient is very small: {grad_mean:.2e}")
                if grad_mean > self.max_threshold:
                    Logger.warning(f"[CheckGradients]  {name} gradient is very large: {grad_mean:.2e}")
            else:
                Logger.warning(f'[CheckGradients]  {name} has no gradient')

        return True
