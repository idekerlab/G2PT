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

class SpearmanLoss(torch.nn.Module):

    def __init__(self, regularization='l2', regularization_strength=1.0, device='cpu'):
        super(SpearmanLoss, self).__init__()
        self.regularization = regularization
        self.regularization_strength = regularization_strength
        self.device = device

    def corrcoef(self, target, pred):
        # np.corrcoef in torch from @mdo
        # https://forum.numer.ai/t/custom-loss-functions-for-xgboost-using-pytorch/960
        pred_n = pred - pred.mean()
        target_n = target - target.mean()
        pred_n = pred_n / pred_n.norm()
        target_n = target_n / target_n.norm()
        return (pred_n * target_n).sum()


    def forward(self, target, pred):
        # fast_soft_sort uses 1-based indexing, divide by len to compute percentage of rank
        pred = soft_rank(
            pred.cpu(),
            regularization=self.regularization,
            regularization_strength=self.regularization_strength,
        ).to(self.device)
        return self.corrcoef(target, pred / pred.shape[-1])

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
