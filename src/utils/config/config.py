from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence

from .data_config import DatasetConfig, PLINKDatasetConfig, TSVDatasetConfig, create_dataset_config
from .model_config import ModelConfig
from .train_config import TrainerConfig, RuntimeConfig
from .utils import _as_namespace


@dataclass(frozen=True)
class SNP2PConfig:
    dataset: DatasetConfig
    model: ModelConfig
    trainer: TrainerConfig
    runtime: RuntimeConfig

    @classmethod
    def from_namespace(cls, args: SimpleNamespace) -> "SNP2PConfig":
        return cls(
            dataset=create_dataset_config(args),
            model=ModelConfig.from_namespace(args),
            trainer=TrainerConfig.from_namespace(args),
            runtime=RuntimeConfig.from_namespace(args),
        )

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SNP2PConfig":
        if {"dataset", "model", "trainer", "runtime"}.issubset(data.keys()):
            return cls(
                dataset=create_dataset_config(**data["dataset"]),
                model=ModelConfig(**data["model"]),
                trainer=TrainerConfig(**data["trainer"]),
                runtime=RuntimeConfig(**data["runtime"]),
            )
        return cls.from_namespace(SimpleNamespace(**data))

    def to_flat_namespace(self) -> SimpleNamespace:
        data: dict[str, Any] = {}
        for section in (self.dataset, self.model, self.trainer, self.runtime):
            data.update(section.__dict__)
        return SimpleNamespace(**data)


def resolve_checkpoint_args(checkpoint: Mapping[str, Any]) -> SimpleNamespace:
    config = checkpoint.get("config")
    if isinstance(config, SNP2PConfig):
        return config.to_flat_namespace()
    if isinstance(config, Mapping):
        return SNP2PConfig.from_mapping(config).to_flat_namespace()

    args = checkpoint.get("arguments")
    if isinstance(args, SNP2PConfig):
        return args.to_flat_namespace()
    return _as_namespace(args)