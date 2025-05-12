import torch
import torch.nn as nn


class MoEHeadPrediction(nn.Module):
    """
    Mixture-of-Experts head that produces a **single real-valued output per
    position** in a sequence of length L.
      h: [B, L, H]  →  preds: [B, L]        (or [B, L, 1] if you keep p=1)
    """
    def __init__(self, hid: int, k_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.k = k_experts
        self.p = 1                  # one scalar per position
        self.top_k = top_k

        # experts produce K scalars, gate picks the mixture weights
        self.experts = nn.Linear(hid, self.p * self.k, bias=True)
        self.gate    = nn.Linear(hid, self.k,            bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, L, H]  — trunk / token embeddings
        returns:
            [B, L]     if p == 1          (default)
            [B, L, p]  otherwise
        """
        # -------- gating ----------
        gate_scores = self.gate(h)                    # [B, L, K]

        if self.top_k is None or self.top_k >= self.k:
            # full softmax over all experts
            weights = torch.softmax(gate_scores, dim=-1)            # [B, L, K]
        else:
            # Switch-Transformer-style top-k sparse softmax
            top_vals, top_idx = gate_scores.topk(self.top_k, dim=-1)    # [B, L, top_k]
            weights = torch.zeros_like(gate_scores)                     # [B, L, K]
            weights.scatter_(-1, top_idx,
                             torch.softmax(top_vals, dim=-1))           # fill only the top-k

        # -------- experts ----------
        expert_out = self.experts(h)                    # [B, L, p*K]
        expert_out = expert_out.view(*h.shape[:2],      # → [B, L, K, p]
                                     self.k, self.p)

        # weighted sum over K experts
        preds = (weights.unsqueeze(-1) * expert_out).sum(dim=2)  # [B, L, p]

        # squeeze the singleton phenotype dimension if p == 1
        if self.p == 1:
            preds = preds.squeeze(-1)                   # [B, L]

        return preds