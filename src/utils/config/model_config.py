from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class ModelConfig:
    onto: Optional[str] = None
    snp2gene: Optional[str] = None
    interaction_types: Sequence[str] = field(default_factory=lambda: ["default"])
    sys2pheno: bool = True
    gene2pheno: bool = False
    snp2pheno: bool = False
    sys2env: bool = False
    env2sys: bool = False
    sys2gene: bool = False
    dense_attention: bool = False
    use_sparse_attention: bool = True
    snp2gene_gating: bool = False
    snp2gene_null_gate: bool = False
    snp2gene_attention_dropout: Optional[float] = None
    snp2gene_null_logit_init: float = -10.0
    snp2gene_source_prior: bool = False
    snp2gene_source_prior_init: float = 0.0
    snp_shallow_predictor: bool = False
    snp_shallow_feed_g2pt: bool = False
    snp_shallow_layers: int = 1
    snp_shallow_alpha_init: float = 1.0
    g2pt_residual_alpha_init: float = 1.0
    prediction_activation: str = "tanh"
    use_block_embedding: Optional[bool] = None
    block_bias: bool = False
    use_gating: bool = False  # Enable phenotype-conditioned gating
    gate_hidden_dim: int = 64  # Hidden dimension for gate MLPs
    gate_temperature: float = 1.0  # Temperature for gate softmax
    gate_sparsity_weight: float = 0.01  # Weight for sparsity regularization
    gate_entropy_weight: float = 0.001  # Weight for entropy regularization
    gate_coherence_weight: float = 0.0  # Weight for pathway coherence
    hidden_dims: int = 256
    n_heads: int = 4
    prediction_head: int = 1
    dropout: float = 0.2
    cov_effect: str = "pre"
    use_hierarchical_transformer: bool = False
    use_moe: bool = False
    independent_predictors: bool = False

    @classmethod
    def from_namespace(cls, args: SimpleNamespace) -> "ModelConfig":
        return cls(
            onto=getattr(args, "onto", None),
            snp2gene=getattr(args, "snp2gene", None),
            interaction_types=list(getattr(args, "interaction_types", ["default"])),
            sys2pheno=bool(getattr(args, "sys2pheno", True)),
            gene2pheno=bool(getattr(args, "gene2pheno", False)),
            snp2pheno=bool(getattr(args, "snp2pheno", False)),
            sys2env=bool(getattr(args, "sys2env", False)),
            env2sys=bool(getattr(args, "env2sys", False)),
            sys2gene=bool(getattr(args, "sys2gene", False)),
            dense_attention=bool(getattr(args, "dense_attention", False)),
            use_sparse_attention=bool(getattr(args, "use_sparse_attention", True)),
            snp2gene_gating=bool(getattr(args, "snp2gene_gating", False)),
            snp2gene_null_gate=bool(getattr(args, "snp2gene_null_gate", False)),
            snp2gene_attention_dropout=getattr(args, "snp2gene_attention_dropout", None),
            snp2gene_null_logit_init=float(getattr(args, "snp2gene_null_logit_init", -10.0)),
            snp2gene_source_prior=bool(getattr(args, "snp2gene_source_prior", False)),
            snp2gene_source_prior_init=float(getattr(args, "snp2gene_source_prior_init", 0.0)),
            snp_shallow_predictor=bool(getattr(args, "snp_shallow_predictor", False)),
            snp_shallow_feed_g2pt=bool(getattr(args, "snp_shallow_feed_g2pt", False)),
            snp_shallow_layers=int(getattr(args, "snp_shallow_layers", 1)),
            snp_shallow_alpha_init=float(getattr(args, "snp_shallow_alpha_init", 1.0)),
            g2pt_residual_alpha_init=float(getattr(args, "g2pt_residual_alpha_init", 1.0)),
            prediction_activation=getattr(args, "prediction_activation", "tanh"),
            use_block_embedding=getattr(args, "use_block_embedding", None),
            block_bias=bool(getattr(args, "block_bias", False)),
            use_gating=bool(getattr(args, "use_gating", False)),
            gate_hidden_dim=int(getattr(args, "gate_hidden_dim", 64)),
            gate_temperature=float(getattr(args, "gate_temperature", 1.0)),
            gate_sparsity_weight=float(getattr(args, "gate_sparsity_weight", 0.01)),
            gate_entropy_weight=float(getattr(args, "gate_entropy_weight", 0.001)),
            gate_coherence_weight=float(getattr(args, "gate_coherence_weight", 0.0)),
            hidden_dims=int(getattr(args, "hidden_dims", 256)),
            n_heads=int(getattr(args, "n_heads", 4)),
            prediction_head=int(getattr(args, "prediction_head", 1)),
            dropout=float(getattr(args, "dropout", 0.2)),
            cov_effect=getattr(args, "cov_effect", "pre"),
            use_hierarchical_transformer=bool(getattr(args, "use_hierarchical_transformer", False)),
            use_moe=bool(getattr(args, "use_moe", False)),
            independent_predictors=bool(getattr(args, "independent_predictors", False)),
        )
