from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 300
    lr: float = 0.001
    wd: float = 0.001
    z_weight: float = 1.0
    batch_size: int = 128
    val_step: int = 20
    patience: int = 10
    jobs: int = 0
    loss: str = "default"
    focal_loss_alpha: float = 0.25
    focal_loss_gamma: float = 2.0
    label_smoothing: float = 0.0

    @classmethod
    def from_namespace(cls, args: SimpleNamespace) -> "TrainerConfig":
        return cls(
            epochs=int(getattr(args, "epochs", 300)),
            lr=float(getattr(args, "lr", 0.001)),
            wd=float(getattr(args, "wd", 0.001)),
            z_weight=float(getattr(args, "z_weight", 1.0)),
            batch_size=int(getattr(args, "batch_size", 128)),
            val_step=int(getattr(args, "val_step", 20)),
            patience=int(getattr(args, "patience", 10)),
            jobs=int(getattr(args, "jobs", 0)),
            loss=getattr(args, "loss", "default"),
            focal_loss_alpha=float(getattr(args, "focal_loss_alpha", 0.25)),
            focal_loss_gamma=float(getattr(args, "focal_loss_gamma", 2.0)),
            label_smoothing=float(getattr(args, "label_smoothing", 0.0)),
        )


@dataclass(frozen=True)
class RuntimeConfig:
    cuda: Optional[int] = None
    mlm: bool = False
    regression: bool = False
    n_cov: int = 4
    target_phenotype: Optional[str] = None

    @classmethod
    def from_namespace(cls, args: SimpleNamespace) -> "RuntimeConfig":
        return cls(
            cuda=getattr(args, "cuda", None),
            mlm=bool(getattr(args, "mlm", False)),
            regression=bool(getattr(args, "regression", False)),
            n_cov=int(getattr(args, "n_cov", 4)),
            target_phenotype=getattr(args, "target_phenotype", None),
        )