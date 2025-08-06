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
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

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


class MultiplePhenotypeLoss(nn.Module):
    """
    Custom loss function for handling multiple loss types based on column indices.

    Attributes:
        bce_cols (list): Column indices for which Binary Cross Entropy loss is applied.
        mse_cols (list): Column indices for which Mean Squared Error loss is applied.
    """

    def __init__(self, bce_cols, mse_cols, label_smoothing=0.0):
        """
        Initializes the CustomLoss.

        Parameters:
            bce_cols (list): List of column indices to use Binary Cross Entropy loss.
            mse_cols (list): List of column indices to use Mean Squared Error loss.
        """
        super(MultiplePhenotypeLoss, self).__init__()
        self.bce_cols = bce_cols
        self.mse_cols = mse_cols
        self.n_tasks = len(bce_cols) + len(mse_cols)
        self.log_vars = nn.Parameter(torch.zeros(self.n_tasks))
        self.label_smoothing = label_smoothing
        if label_smoothing > 0:
            self.bce_loss = BCEWithLogitsLossWithLabelSmoothing(alpha=label_smoothing)
        else:
            self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, predictions, targets):
        """
        Compute the loss over multiple columns using either BCE or MSE for valid targets.
        Invalid target values marked as -9 are ignored.

        Parameters:
            predictions (torch.Tensor): Tensor with shape (batch_size, n_columns).
            targets (torch.Tensor): Tensor with the same shape as predictions.

        Returns:
            torch.Tensor: The computed average loss.
        """
        total_loss = 0.0
        loss_count = 0
        #print(predictions.size(), targets.size())
        # Process columns with Binary Cross Entropy loss
        for col in self.bce_cols:
            pred = predictions[:, col]
            target = targets[:, col]
            s = self.log_vars[col]
            # Skip invalid targets (-9)
            valid_mask = (target != -9)
            if valid_mask.sum() > 0:
                # F.binary_cross_entropy_with_logits expects raw logits as input.
                loss = self.bce_loss(pred[valid_mask], target[valid_mask].float())
                loss = torch.exp(-s) * loss + s
                total_loss += loss * 2
                loss_count += 1

        # Process columns with Mean Squared Error loss
        for col in self.mse_cols:
            pred = predictions[:, col]
            target = targets[:, col]
            s = self.log_vars[col]
            valid_mask = (target != -9)
            if valid_mask.sum() > 0:
                loss = F.mse_loss(pred[valid_mask], target[valid_mask])
                loss = 0.5 * (torch.exp(-s) * loss + s)
                total_loss += loss
                loss_count += 1

        return total_loss / loss_count if loss_count > 0 else total_loss


class BCEWithLogitsLossWithLabelSmoothing(nn.Module):
    def __init__(self, alpha=0.1, reduction='mean'):
        super(BCEWithLogitsLossWithLabelSmoothing, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        smoothed_labels = targets * (1.0 - self.alpha) + 0.5 * self.alpha
        loss = F.binary_cross_entropy_with_logits(inputs, smoothed_labels, reduction=self.reduction)
        return loss