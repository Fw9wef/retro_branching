import torch
import torch.nn.functional as F

class MeanSquaredError:
    def __init__(self,
                 reduction='mean'):
        self.reduction = reduction

    def extract(self, _input, target, reduction='default'):
        if reduction == 'default':
            reduction = self.reduction
        if reduction == 'clip':
            reduction = 'none'
            loss = F.mse_loss(_input, target, reduction=reduction)
            loss = torch.clip(loss, max=1)
            return torch.mean(loss)
        return F.mse_loss(_input, target, reduction=reduction)