import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, in_cov:int, hid:int):
        super().__init__()
        self.to_gamma = nn.Linear(in_cov, hid, bias=True)
        self.to_beta  = nn.Linear(in_cov, hid, bias=True)

    def forward(self, x, cov):
        """
        x   : [B, N, H]  – tokens (gene / system / etc.)
        cov : [B, C]     – covariate vector
        """
        gamma = self.to_gamma(cov).unsqueeze(1)   # [B, 1, H]
        beta  = self.to_beta(cov).unsqueeze(1)
        return gamma * x + beta