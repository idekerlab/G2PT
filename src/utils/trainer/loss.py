import torch

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