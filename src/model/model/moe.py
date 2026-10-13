import torch
import torch.nn as nn


class MoEHeadPrediction(nn.Module):
    """
    Mixture-of-Experts head producing real-valued outputs for P phenotypes.
    Uses "Top-2 gating" (Switch Transformer style) by default.
    """
    def __init__(self, hid: int, n_pheno: int, k_experts: int = 8,
                 top_k: int = 2):
        super().__init__()
        self.experts = nn.Linear(hid, n_pheno * k_experts, bias=True)
        self.gate    = nn.Linear(hid, k_experts, bias=False)
        self.k       = k_experts
        self.p       = n_pheno
        self.top_k   = top_k

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, H] trunk embeddings
        returns: [B, P] real-valued predictions
        """
        B = h.size(0)

        # -------- gating ----------
        gate_scores = self.gate(h)                       # [B, K]
        if self.top_k is None or self.top_k >= self.k:   # full softmax
            weights = torch.softmax(gate_scores, dim=-1) # [B, K]
        else:                                            # top-k sparse softmax
            top_vals, top_idx = gate_scores.topk(self.top_k, dim=-1)
            weights = torch.zeros_like(gate_scores)      # [B, K]
            weights.scatter_(1, top_idx, torch.softmax(top_vals, dim=-1))

        # -------- experts ----------
        expert_out = self.experts(h)                     # [B, P*K]
        expert_out = expert_out.view(B, self.k, self.p)  # [B, K, P]

        # weighted sum over experts
        preds = (weights.unsqueeze(-1) * expert_out).sum(dim=1)  # [B, P]
        return preds