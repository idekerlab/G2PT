from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Mapping, Optional, Sequence, Union


@dataclass
class DatasetConfig:
    train_cov: Optional[str] = None
    train_pheno: Optional[str] = None
    val_cov: Optional[str] = None
    val_pheno: Optional[str] = None
    test_cov: Optional[str] = None
    test_pheno: Optional[str] = None
    cov_ids: Sequence[str] = field(default_factory=list)
    pheno_ids: Sequence[str] = field(default_factory=list)
    bt: Sequence[str] = field(default_factory=list)
    qt: Sequence[str] = field(default_factory=list)
    flip: bool = False
    input_format: str = "indices"
    subsample: Optional[int] = None

    @classmethod
    def from_namespace(cls, args: SimpleNamespace):
        field_names = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: getattr(args, k, None) for k in field_names})


class PLINKDatasetConfig(DatasetConfig):
    train_bfile: Optional[str] = None
    val_bfile: Optional[str] = None
    test_bfile: Optional[str] = None
    input_format = 'plink'


class TSVDatasetConfig(DatasetConfig):
    train_tsv: Optional[str] = None
    val_tsv: Optional[str] = None
    test_tsv: Optional[str] = None
    input_format = 'tsv'


def create_dataset_config(source: Union[SimpleNamespace, dict]) -> DatasetConfig:
    # 1. 포맷 확인 (source가 dict인지 namespace인지에 따라 접근법 분기)
    if isinstance(source, dict):
        fmt = source.get("input_format", "indices")
    else:
        fmt = getattr(source, "input_format", "indices")

    # 2. 포맷에 따른 클래스 매핑
    config_map = {
        "plink": PLINKDatasetConfig,
        "bed": PLINKDatasetConfig,
        "tsv": TSVDatasetConfig,
        "indices": DatasetConfig,
    }

    target_cls = config_map.get(fmt, DatasetConfig)

    # 3. 해당 클래스의 from_namespace 또는 생성자 호출
    if isinstance(source, dict):
        valid_keys = target_cls.__dataclass_fields__.keys()
        filtered_data = {k: v for k, v in source.items() if k in valid_keys}
        return target_cls(**filtered_data)
    else:
        # namespace인 경우
        return target_cls.from_namespace(source)
