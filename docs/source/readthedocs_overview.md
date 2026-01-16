# Overview

G2PT is a hierarchical Genotype-to-Phenotype Transformer that models
information flow from SNPs to genes, systems, and phenotypes. The
Read the Docs pages focus on how to run experiments and navigate the API
reference.

## Quickstart

1. Create the environment and make the repo importable:

```bash
conda env create -f environment.yml
conda activate G2PT_github
export PYTHONPATH=.
```

2. Prepare inputs:
   - Genotypes: PLINK `.bed/.bim/.fam` or TSV matrices.
   - Covariates/phenotypes: tab-delimited files with `FID` and `IID` columns.
   - Ontology (`--onto`) and SNP-to-gene mapping (`--snp2gene`).

3. Train a model (PLINK example):

```bash
python train_snp2p_model.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --train-bfile /path/to/train \
  --train-cov /path/to/train.cov --train-pheno /path/to/train.pheno \
  --val-bfile /path/to/val \
  --val-cov /path/to/val.cov --val-pheno /path/to/val.pheno \
  --bt PHENOTYPE \
  --out outputs/run1
```

4. Generate predictions and attention summaries:

```bash
python predict_attention.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --bfile /path/to/test \
  --cov /path/to/test.cov --pheno /path/to/test.pheno \
  --model outputs/run1/model_best.pth \
  --out outputs/run1/test
```

## Where to go next

- API reference pages for model, dataset, and trainer components are under the
  **API Documentation** section.
- For full CLI usage details and additional examples, see the repository
  [README](../README.md).
