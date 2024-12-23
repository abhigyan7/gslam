from abc import ABC

from .map import GaussianSplattingData
from typing import Dict

from torch.optim import Optimizer
import torch


class PruningStrategy(ABC):
    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        return


class PruneLowOpacity(PruningStrategy):
    def __init__(self, min_opacity: float):
        self.min_opacity = min_opacity

    def step(self, splats: GaussianSplattingData, optimizers: Dict[str, Optimizer]):
        parameters = splats.named_parameters()

        opacities = torch.sigmoid(splats.opacities)

        keep_mask = opacities > self.min_opacity

        for parameter_name, parameter in parameters:
            parameter.data = parameter[keep_mask].data
            updated_parameter = parameter[keep_mask].detach()

            optimizer = optimizers[parameter_name]

            for i in range(len(optimizer.param_groups)):
                parameter_state = optimizer.state[parameter]
                del optimizer.state[parameter]
                for key in parameter_state.keys():
                    if key == 'step':
                        continue
                    v = parameter_state[key]
                    parameter_state[key] = v[keep_mask]
                optimizer.param_groups[i]['params'] = [updated_parameter]
                optimizer.state[updated_parameter] = parameter_state

        return
