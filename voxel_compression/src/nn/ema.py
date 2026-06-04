import copy
import torch
from torch import nn


class EMA(nn.Module):
    def __init__(self, model: nn.Module, decay: float = 0.999):
        super().__init__()
        self.decay = decay
        self.model = model
        self.shadow = copy.deepcopy(model)
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self):
        for p, s in zip(self.model.parameters(), self.shadow.parameters()):
            s.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    def forward(self, *args, **kwargs):
        return self.shadow(*args, **kwargs)
