import torch
import torch.nn as nn
import torch.nn.functional as F


class PhenotypeConditionedGate(nn.Module):
    """
    Phenotype-conditioned gate for dynamic hierarchy.

    Learns to control information flow based on phenotype embedding.
    Output is sigmoid-activated ∈ [0, 1] for interpretability.

    Architecture:
        phenotype_embedding → Linear → ReLU → Linear → Sigmoid → gate
    """

    def __init__(self, phenotype_dim, hidden_dim=64, temperature=1.0):
        """
        Args:
            phenotype_dim: Dimension of phenotype embedding
            hidden_dim: Hidden layer dimension for gate MLP
            temperature: Softmax temperature (lower = more discrete)
        """
        super().__init__()
        self.temperature = temperature

        # 2-layer MLP for gate computation
        self.gate_mlp = nn.Sequential(
            nn.Linear(phenotype_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output ∈ [0, 1]
        )

    def forward(self, phenotype_embedding):
        """
        Args:
            phenotype_embedding: (batch_size, phenotype_dim)

        Returns:
            gate: (batch_size, 1) values in [0, 1]
        """
        gate = self.gate_mlp(phenotype_embedding)

        # Apply temperature scaling
        if self.training and self.temperature != 1.0:
            gate = torch.sigmoid((torch.logit(gate.clamp(1e-7, 1-1e-7)) / self.temperature))

        return gate

    def get_gate_stats(self, phenotype_embedding):
        """Compute gate statistics for analysis."""
        with torch.no_grad():
            gate = self.forward(phenotype_embedding)
            return {
                'mean': gate.mean().item(),
                'std': gate.std().item(),
                'min': gate.min().item(),
                'max': gate.max().item(),
                'sparsity': (gate < 0.1).float().mean().item()  # % gates < 0.1
            }
