import torch
import torch.nn as nn
import torch.nn.functional as F


class GateRegularizationLoss(nn.Module):
    """
    Regularization losses for phenotype-conditioned gates.

    Three components:
    1. Sparsity loss: Encourage gates to be near 0 or 1
    2. Entropy loss: Encourage diverse gate usage across batch
    3. Pathway coherence loss: Encourage related pathways to have similar gates
    """

    def __init__(self, sparsity_weight=0.01, entropy_weight=0.001, coherence_weight=0.0):
        """
        Args:
            sparsity_weight: Weight for L1 sparsity loss
            entropy_weight: Weight for entropy loss
            coherence_weight: Weight for pathway coherence loss
        """
        super().__init__()
        self.sparsity_weight = sparsity_weight
        self.entropy_weight = entropy_weight
        self.coherence_weight = coherence_weight

    def sparsity_loss(self, gates):
        """
        L1 sparsity loss to encourage gates near 0 or 1.

        Args:
            gates: Dict of gate tensors (B, 1) for each propagation step

        Returns:
            Scalar loss
        """
        total_loss = 0.0
        for gate_name, gate_values in gates.items():
            # L1 loss encourages sparsity
            total_loss += gate_values.abs().mean()
        return total_loss / len(gates)

    def entropy_loss(self, gates):
        """
        Entropy loss to encourage diverse gate usage across batch.

        High entropy = gates vary across samples (good)
        Low entropy = all samples use same gates (bad)

        Args:
            gates: Dict of gate tensors (B, 1)

        Returns:
            Negative entropy (minimize to maximize entropy)
        """
        total_entropy = 0.0
        for gate_name, gate_values in gates.items():
            # Compute entropy of gate distribution
            p = gate_values.squeeze(-1)  # (B,)
            # Binary entropy: H = -p*log(p) - (1-p)*log(1-p)
            eps = 1e-7
            entropy = -(p * torch.log(p + eps) + (1 - p) * torch.log(1 - p + eps))
            total_entropy += entropy.mean()

        # Return negative entropy (minimize to maximize)
        return -total_entropy / len(gates)

    def pathway_coherence_loss(self, gates, ontology_hierarchy=None):
        """
        Pathway coherence loss to encourage related pathways to have similar gates.

        Args:
            gates: Dict of gate tensors
            ontology_hierarchy: Optional hierarchy structure (not implemented yet)

        Returns:
            Scalar loss
        """
        # TODO: Implement if ontology hierarchy is provided
        # For now, just return 0
        return torch.tensor(0.0, device=list(gates.values())[0].device)

    def forward(self, gates):
        """
        Compute total gate regularization loss.

        Args:
            gates: Dict of gate tensors from model.get_gate_values()

        Returns:
            total_loss: Weighted sum of all regularization terms
            loss_dict: Dict of individual loss components for logging
        """
        if not gates:
            return torch.tensor(0.0), {}

        loss_dict = {}

        # Sparsity loss
        if self.sparsity_weight > 0:
            sparsity = self.sparsity_loss(gates)
            loss_dict['gate_sparsity'] = sparsity.item()
        else:
            sparsity = torch.tensor(0.0, device=list(gates.values())[0].device)

        # Entropy loss
        if self.entropy_weight > 0:
            entropy = self.entropy_loss(gates)
            loss_dict['gate_entropy'] = entropy.item()
        else:
            entropy = torch.tensor(0.0, device=list(gates.values())[0].device)

        # Pathway coherence loss
        if self.coherence_weight > 0:
            coherence = self.pathway_coherence_loss(gates)
            loss_dict['gate_coherence'] = coherence.item()
        else:
            coherence = torch.tensor(0.0, device=list(gates.values())[0].device)

        # Total weighted loss
        total_loss = (self.sparsity_weight * sparsity +
                     self.entropy_weight * entropy +
                     self.coherence_weight * coherence)

        loss_dict['gate_total'] = total_loss.item()

        return total_loss, loss_dict
