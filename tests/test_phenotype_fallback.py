"""Phenotype fallback regressions using real Torch, sgkit and PLINK files.

Port of the regression scenarios contributed by sushan5140 in
https://github.com/GIST-CSBL/G2PT/pull/1 (30472076b73573a116a7e423fc729cd931c87819).
PLINK files are tiny generated fixtures. Only the tree parser is a test double.
Run from the repository root with python -m pytest -q tests/test_phenotype_fallback.py.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from bed_reader import to_bed
from sgkit.io import plink
from torch.utils.data import DataLoader

from src.utils.data.dataset import GenotypeDataset, PLINKDataset, SNP2PCollator, TSVDataset
from src.utils.trainer.loss import MultiplePhenotypeLoss


IIDS = ["p3", "p1", "p2", "p6", "p4", "p5"]
COUNTS = np.array([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]], dtype=float)


@pytest.fixture
def inputs(tmp_path):
    prefix = tmp_path / "tiny"
    to_bed(str(prefix) + ".bed", COUNTS, count_A1=False, properties={
        "fid": ["f" + iid[1:] for iid in IIDS], "iid": IIDS,
        "sid": ["101", "102"], "chromosome": ["1", "1"],
        "bp_position": [10, 20], "allele_1": ["A", "C"], "allele_2": ["G", "T"],
        "sex": [1, 2, 0, 1, 2, 0], "pheno": [1, 2, 0, -9, 1, 2],
    })
    pd.DataFrame(COUNTS, index=IIDS, columns=[101, 102]).to_csv(tmp_path / "geno.tsv", sep="\t")
    cov = pd.DataFrame({"FID": ["f" + iid[1:] for iid in IIDS], "IID": IIDS,
                        "SEX": [0, 1, -9, 0, 1, -9], "AGE": [20, 30, 40, 50, 60, 70],
                        "PHENOTYPE": [1., 0., np.nan, 0., 1., 0.]})
    cov = cov.iloc[[2, 5, 0, 4, 1, 3]].reset_index(drop=True)
    cov.to_csv(tmp_path / "cov.tsv", sep="\t", index=False)
    return tmp_path, cov


def tree(kind):
    snps = [101, 102] if kind == "tsv" else ["101", "102"]
    return SimpleNamespace(n_snps=2, n_genes=1, n_systems=1, block=False,
                           snp2ind=dict(zip(snps, range(2))),
                           set_snp2chr=lambda value: None, set_snp2pos=lambda value: None)


def make_dataset(path, kind="plink", **kwargs):
    kwargs.setdefault("cov", path / "cov.tsv")
    if kind == "base":
        return GenotypeDataset(tree(kind), **kwargs)
    if kind == "tsv":
        return TSVDataset(tree(kind), path / "geno.tsv", **kwargs)
    return PLINKDataset(tree(kind), str(path / "tiny"), **kwargs)


def set_fam(path, values):
    fam = pd.read_csv(path / "tiny.fam", sep=r"\s+", header=None)
    fam[5] = values
    fam.to_csv(path / "tiny.fam", sep=" ", index=False, header=False, na_rep="NA")


def assert_aligned(dataset, expected):
    assert dataset.cov_df.IID.tolist() == dataset.pheno_df.IID.tolist()
    for index, row in dataset.cov_df.iterrows():
        item = dataset[index]
        torch.testing.assert_close(item["phenotype"], torch.tensor(expected[row.IID], dtype=torch.float32))
        assert item["phenotype_indices"].tolist() == list(range(dataset.n_pheno))
        assert item["covariates"].shape == (dataset.n_cov,)
        if hasattr(dataset, "ind2iid"):
            assert dataset.ind2iid[index] == row.IID
            # Independent expected SNP indices under the existing default allele flip.
            counts = COUNTS[IIDS.index(row.IID)]
            assert item["genotype"]["snp"][:2].tolist() == ((2 - counts) * 2 + [0, 1]).tolist()


def assert_metadata(dataset, ids, binary=()):
    assert list(dataset.pheno_ids) == ids
    assert dataset.n_pheno == len(ids)
    assert dataset.has_phenotype
    assert dataset.pheno_range == list(range(len(ids)))
    assert dataset.pheno2ind == {name: i for i, name in enumerate(ids)}
    assert dataset.ind2pheno == dict(enumerate(ids))
    assert dataset.bt_inds == [ids.index(name) for name in binary]
    assert dataset.qt_inds == [i for i, name in enumerate(ids) if name not in binary]
    assert dataset.pheno2type == {name: "bt" if name in binary else "qt" for name in ids}


@pytest.mark.parametrize("kind", ["base", "tsv", "plink"])
@pytest.mark.parametrize("binary", [False, True])
def test_covariate_fallback(inputs, kind, binary):
    path, cov = inputs
    kwargs = {"bt": ["PHENOTYPE"]} if binary else {}
    dataset = make_dataset(path, kind, **kwargs)
    assert_metadata(dataset, ["PHENOTYPE"], ["PHENOTYPE"] if binary else [])
    expected = {row.IID: [row.PHENOTYPE] for row in cov.fillna(-9).itertuples()}
    assert_aligned(dataset, expected)
    assert pd.read_csv(path / "cov.tsv", sep="\t").PHENOTYPE.isna().sum() == 1


@pytest.mark.parametrize("kind", ["base", "tsv", "plink"])
@pytest.mark.parametrize("explicit_order", [False, True])
def test_separate_file_precedence_and_multiple_tasks(inputs, kind, explicit_order):
    path, cov = inputs
    pheno = cov[["FID", "IID"]].iloc[::-1].copy()
    pheno["QUANT"] = [-2.5, np.nan, 0, 2.5, 1.25, -9]
    pheno["BINARY"] = [0, 1, 1, 0, np.nan, 1]
    pheno.to_csv(path / "pheno.tsv", sep="\t", index=False)
    ids = ["BINARY", "QUANT"] if explicit_order else ["QUANT", "BINARY"]
    dataset = make_dataset(path, kind, pheno=path / "pheno.tsv",
                           pheno_ids=ids if explicit_order else (), bt=["BINARY"], qt=["QUANT"])
    assert_metadata(dataset, ids, ["BINARY"])
    assert_aligned(dataset, pheno.set_index("IID")[ids].fillna(-9).T.to_dict("list"))


@pytest.mark.parametrize("external_cov", [False, True])
@pytest.mark.parametrize("task", ["binary", "quantitative", "default"])
def test_raw_fam_values_and_sex(inputs, external_cov, task):
    path, cov = inputs
    cov.drop(columns="PHENOTYPE").to_csv(path / "cov.tsv", sep="\t", index=False)
    if task == "binary":
        set_fam(path, [1, 2, 0, -9, np.nan, 2])
        expected = [0, 1, -9, -9, -9, 1]
        kwargs = {"bt": ["PHENOTYPE"]}
    else:
        set_fam(path, [-2.5, 0, -1, -9, np.nan, 1.25])
        expected = [-2.5, 0, -1, -9, -9, 1.25]
        kwargs = {"qt": ["PHENOTYPE"]} if task == "quantitative" else {}
    dataset = make_dataset(path, cov=path / "cov.tsv" if external_cov else None, **kwargs)
    assert_metadata(dataset, ["PHENOTYPE"], ["PHENOTYPE"] if task == "binary" else [])
    assert list(dataset.ind2iid.values()) == IIDS
    assert_aligned(dataset, {iid: [value] for iid, value in zip(IIDS, expected)})
    if not external_cov:
        assert dataset.cov_ids == ["SEX"] and dataset.n_cov == 2
        assert dataset.cov_mean_dict == {} and dataset.cov_std_dict == {}
        assert [dataset[i]["covariates"].tolist() for i in range(6)] == [
            [1., 0.], [0., 1.], [0., 0.], [1., 0.], [0., 1.], [0., 0.]]


def test_sgkit_decodes_real_fixture(inputs):
    path, _ = inputs
    set_fam(path, [1, 2, 0, -9, 2.5, np.nan])
    data = plink.read_plink(path=str(path / "tiny"))
    np.testing.assert_array_equal(data.call_genotype.values.sum(axis=-1).T, COUNTS)
    # Demonstrates why using the decoded phenotype would destroy quantitative data.
    assert data.sample_phenotype.values.tolist() == [1, 2, -1, -1, -1, -1]


@pytest.mark.parametrize("source", ["cov", "pheno"])
def test_external_phenotype_overrides_missing_fam(inputs, source):
    path, cov = inputs
    set_fam(path, [-9] * 6)
    kwargs = {}
    if source == "pheno":
        cov[["FID", "IID", "PHENOTYPE"]].to_csv(path / "pheno.tsv", sep="\t", index=False)
        kwargs = {"cov": None, "pheno": path / "pheno.tsv"}
    dataset = make_dataset(path, bt=["PHENOTYPE"], **kwargs)
    assert len(dataset) == 6
    assert_aligned(dataset, {row.IID: [row.PHENOTYPE] for row in cov.fillna(-9).itertuples()})


def test_three_way_intersection_preserves_genotype_order(inputs):
    path, cov = inputs
    cov[cov.IID.isin(["p1", "p3", "p4"])].to_csv(path / "cov.tsv", sep="\t", index=False)
    pheno = cov[cov.IID.isin(["p4", "p3", "p2"])][["FID", "IID", "PHENOTYPE"]]
    pheno.to_csv(path / "pheno.tsv", sep="\t", index=False)
    dataset = make_dataset(path, pheno=path / "pheno.tsv")
    assert list(dataset.ind2iid.values()) == ["p3", "p4"]
    assert_aligned(dataset, {"p3": [1], "p4": [1]})


@pytest.mark.parametrize("kind", ["base", "tsv"])
def test_missing_phenotype_source(inputs, kind):
    path, cov = inputs
    cov.drop(columns="PHENOTYPE").to_csv(path / "cov.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="Phenotype data required"):
        make_dataset(path, kind)


@pytest.mark.parametrize("kwargs, message", [
    ({"pheno_ids": ["UNKNOWN"]}, "UNKNOWN"),
    ({"qt": ["UNKNOWN"]}, "UNKNOWN"),
    ({"bt": ["PHENOTYPE"], "qt": ["PHENOTYPE"]}, "both"),
    ({"pheno_ids": ["PHENOTYPE", "PHENOTYPE"]}, "unique"),
])
def test_invalid_task_configuration(inputs, kwargs, message):
    path, _ = inputs
    with pytest.raises(ValueError, match=message):
        make_dataset(path, **kwargs)


@pytest.mark.parametrize("source", ["cov", "pheno"])
@pytest.mark.parametrize("problem", ["duplicate", "missing_id"])
def test_invalid_sample_ids(inputs, source, problem):
    path, cov = inputs
    frame = cov.copy()
    if problem == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    else:
        frame = frame.drop(columns="IID")
    frame.to_csv(path / f"{source}.tsv", sep="\t", index=False)
    kwargs = {"pheno": path / "pheno.tsv"} if source == "pheno" else {}
    with pytest.raises(ValueError, match="IID"):
        make_dataset(path, **kwargs)


def test_binary_fam_rejects_quantitative_values(inputs):
    path, _ = inputs
    set_fam(path, [1, 2, 1.5, -9, 0, 1])
    with pytest.raises(ValueError, match="Binary PLINK"):
        make_dataset(path, cov=None, bt=["PHENOTYPE"])


@pytest.mark.parametrize("binary", [False, True])
def test_dataloader_collator_and_training_loss(inputs, binary):
    path, _ = inputs
    dataset = make_dataset(path, cov=None, bt=["PHENOTYPE"] if binary else ())
    batch = next(iter(DataLoader(dataset, batch_size=6,
                                collate_fn=SNP2PCollator(dataset.tree_parser))))
    assert batch["phenotype"].shape == (6, 1)
    assert batch["covariates"].shape == (6, 2)
    predictions = torch.zeros(6, 1, requires_grad=True)
    criterion = MultiplePhenotypeLoss(dataset.bt_inds, dataset.qt_inds)
    loss = criterion(predictions, batch["phenotype"])
    assert torch.isfinite(loss)
    loss.backward()
    missing = batch["phenotype"] == -9
    assert torch.all(predictions.grad[missing] == 0)
    assert torch.all(torch.isfinite(predictions.grad))


def test_leading_zero_sample_ids(inputs):
    path, cov = inputs
    id_map = dict(zip(IIDS, ["003", "001", "002", "006", "004", "005"]))
    fam = pd.read_csv(path / "tiny.fam", sep=r"\s+", header=None, dtype=str)
    fam[1] = fam[1].map(id_map)
    fam.to_csv(path / "tiny.fam", sep=" ", index=False, header=False)
    cov["IID"] = cov.IID.map(id_map)
    cov.to_csv(path / "cov.tsv", sep="\t", index=False)
    dataset = make_dataset(path)
    assert list(dataset.ind2iid.values()) == list(id_map.values())
    assert dataset.pheno_df.IID.tolist() == list(id_map.values())


def test_no_sample_overlap(inputs):
    path, cov = inputs
    cov["IID"] = "other_" + cov.IID
    cov.to_csv(path / "cov.tsv", sep="\t", index=False)
    with pytest.raises(ValueError, match="No overlapping"):
        make_dataset(path)


def test_validation_covariate_statistics_reused(inputs):
    path, _ = inputs
    train = make_dataset(path)
    validation = make_dataset(path, cov_mean_dict=train.cov_mean_dict,
                              cov_std_dict=train.cov_std_dict,
                              reference_alleles=train.variant_alleles)
    assert validation.cov_mean_dict is train.cov_mean_dict
    assert validation.cov_std_dict is train.cov_std_dict
    for index in range(len(train)):
        torch.testing.assert_close(train[index]["covariates"], validation[index]["covariates"])
        torch.testing.assert_close(train[index]["genotype"]["snp"], validation[index]["genotype"]["snp"])


def test_fam_covariate_selection_error(inputs):
    path, _ = inputs
    with pytest.raises(ValueError, match="SEX"):
        make_dataset(path, cov=None, cov_ids=["SEX", "AGE"])


@pytest.mark.parametrize("kind", ["plink", "tsv"])
def test_training_factory_fallback(inputs, kind):
    from src.utils.config.data_config import DatasetConfig
    from train_snp2p_model import SNP2PDatasetFactory

    path, _ = inputs
    config = DatasetConfig(bt=["PHENOTYPE"], cov_ids=["SEX", "AGE"])
    dataset_path = str(path / "tiny") if kind == "plink" else path / "geno.tsv"
    dataset = SNP2PDatasetFactory.create_dataset(
        tree(kind), dataset_path, kind, config, cov_path=path / "cov.tsv")
    assert_metadata(dataset, ["PHENOTYPE"], ["PHENOTYPE"])
    assert len(dataset) == 6
    assert dataset[0]["phenotype"].shape == (1,)
