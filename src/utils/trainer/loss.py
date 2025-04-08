import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class CCCLoss(torch.nn.Module):

    def __init__(self, eps=1e-6, mean_diff=True):
        super(CCCLoss, self).__init__()
        self.eps = eps
        self.mean_diff = mean_diff

    def forward(self, y_true, y_hat):
        y_true_mean = torch.mean(y_true)
        y_hat_mean = torch.mean(y_hat)
        y_true_var = torch.var(y_true)
        y_hat_var = torch.var(y_hat)
        y_true_std = torch.std(y_true)
        y_hat_std = torch.std(y_hat)
        vx = y_true - torch.mean(y_true)
        vy = y_hat - torch.mean(y_hat)
        if self.mean_diff:
            denom = (y_true_var + y_hat_var + (y_hat_mean - y_true_mean) ** 2)
        else:
            denom = (y_true_var + y_hat_var )
        pcc = torch.sum(vx * vy) / (
                    torch.sqrt(torch.sum(vx ** 2) + self.eps) * torch.sqrt(torch.sum(vy ** 2) + self.eps))
        ccc = (2 * pcc * y_true_std * y_hat_std) / denom

        ccc = 1 - ccc
        return ccc


class FocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True, gamma=1, alpha=1):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, inputs, targets, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = self.alpha * (1-BCE_EXP)**self.gamma * BCE
                       
        return focal_loss

class VarianceLoss(nn.Module):
    """
    Computes variance-matching loss between predictions and targets within each minibatch.
    Encourages the predictions' variance to match the variance of the targets.
    """
    def __init__(self):
        super(VarianceLoss, self).__init__()

    def forward(self, predictions, targets):
        pred_std = torch.std(predictions, unbiased=False)
        target_std = torch.std(targets, unbiased=False)
        loss = (pred_std - target_std).pow(2)
        return loss