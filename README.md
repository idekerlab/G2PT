# G2PT: A genotype-phenotype transformer to assess and explain polygenic risk

## Overview

Genome-wide association studies have linked millions of genetic variants to human phenotypes, but translating this information clinically has been challenged by limited biological interpretability and widespread genetic interactions. G2PT is a hierarchical Genotype-to-Phenotype Transformer that models bidirectional information flow among polymorphisms, genes, molecular systems, and phenotypes. It has been used to predict metabolic traits in the UK Biobank (e.g., diabetes risk and TG/HDL ratio) and to surface pathway-level explanations through attention weights.


![Figure_1](./Figures/Figure_1.jpg)

## Key capabilities

- Hierarchical transformer with SNP → gene → system → phenotype message passing and optional system ↔ environment edges.
- Works with PLINK binary files or tab-delimited genotype matrices; supports multiple phenotypes (binary via `--bt`, quantitative via `--qt`).
- Distributed training via `torchrun`, with early stopping, masked-language-model pretraining (`--mlm`), and mixture-of-experts predictors (`--use_moe`).
- Predict-only and attention-export pipeline for downstream interpretation.
- Companion notebooks for ontology curation, visualization, Sankey plots, and epistasis exploration.

## Environment setup

The repository ships with a conda environment (Python 3.8, CUDA 12.1-compatible PyTorch nightly) that includes all required dependencies.

```bash
conda env create -f environment.yml
conda activate G2PT_github
# From the repository root so `src/` is importable
export PYTHONPATH=.
```

## Input preparation

1. **Genotypes**
   - PLINK binary files (`.bed/.bim/.fam`) passed via `--train-bfile` / `--val-bfile` / `--test-bfile`.
   - Tab-delimited genotype matrices (rows = IID, columns = variant IDs) passed via `--train-tsv` / `--val-tsv`; set `--input-format` to `indices` or `binary` as appropriate. Use `--flip` if alleles need to be swapped.

2. **Covariates and phenotypes**
   - Tab-separated text matching PLINK `.cov` / `.pheno` conventions. Provide columns for `FID` and `IID` plus any covariates/phenotypes.
   - Restrict covariates with `--cov-ids SEX AGE PC1 PC2 ...`. Declare phenotype types with `--bt` (binary) and `--qt` (quantitative). If a phenotype file is omitted, include a `PHENOTYPE` column in the covariate file.

   *Example covariates* ([samples/train.cov](samples/train.cov)):

   | FID      | IID      | SEX | AGE | PC1 | PC2 | ... | PC10 |
   |----------|----------|-----|-----|-----|-----|-----|------|
   | 10008090 | 10008090 | 1   | 48  | 3   | 0.3 | ... | 0.5  |

   *Example phenotypes* ([samples/train.pheno](samples/train.pheno)):

   | FID      | IID      | PHENOTYPE |
   |----------|----------|-----------|
   | 10008090 | 10008090 | 1.2       |

3. **Ontology / hierarchy** (`--onto`)
   - Tab-delimited file with three columns: parent term, child term (term or gene), and `interaction_type` (e.g., `default` for term→term edges or `gene` for term→gene annotations). For nested subtrees, supply custom interaction types and pass them through `--interaction-types`.

   *Example ontology* ([samples/ontology.txt](samples/ontology.txt)):

   | parent     | child      | interaction_type |
   |------------|------------|------------------|
   | GO:0045834 | GO:0045923 | default          |
   | GO:0045834 | GO:0043552 | default          |
   | GO:0045923 | AKT2       | gene             |
   | GO:0045923 | IL1B       | gene             |
   | GO:0043552 | PIK3R4     | gene             |

4. **SNP-to-gene mapping** (`--snp2gene`)
   - Tab-delimited mapping of SNP IDs to genes. Optional columns such as `chr`, `pos`, or `block_ind` are ingested when present. PLINK `.bim` information overrides overlapping fields.

   *Example mapping* ([samples/snp2gene.txt](samples/snp2gene.txt)):

   | snp              | gene  | chr | pos       |
   |------------------|-------|-----|-----------|
   | 16:56995236:A:C  | CETP  | 16  | 56995236  |
   | 8:126482077:G:A  | TRIB1 | 8   | 126482077 |
   | 19:45416178:T:G  | APOC1 | 19  | 45416178  |
   | 2:27752463:A:G   | GCKR  | 2   | 27752463  |

If you want to collapse a Gene Ontology file using GWAS summary statistics, start with the notebooks in the [G2PT pipeline](#g2pt-pipeline-in-overall).

## Training

> Sample data in `samples/` is randomly generated and only demonstrates the CLI; it will not yield meaningful biological results.

### Single-GPU example (PLINK input)

```bash
python train_snp2p_model.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --train-bfile /path/to/train \
  --train-cov /path/to/train.cov --train-pheno /path/to/train.pheno \
  --val-bfile /path/to/val \
  --val-cov /path/to/val.cov --val-pheno /path/to/val.pheno \
  --bt PHENOTYPE \
  --cov-ids SEX AGE PC1 PC2 PC3 \
  --epochs 50 --batch-size 128 --val-step 20 --patience 10 \
  --hidden-dims 256 --lr 1e-3 --wd 1e-3 --dropout 0.2 \
  --sys2env --env2sys --sys2gene \
  --out outputs/run1
```

### TSV genotype example

```bash
python train_snp2p_model.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --train-tsv /path/to/train_genotypes.tsv \
  --train-cov /path/to/train.cov --train-pheno /path/to/train.pheno \
  --val-tsv /path/to/val_genotypes.tsv \
  --val-cov /path/to/val.cov --val-pheno /path/to/val.pheno \
  --bt PHENOTYPE --input-format indices \
  --out outputs/run_tsv
```

### Multi-GPU (distributed data parallel)

Use `torchrun` to launch one process per GPU. Batch size and worker counts should be tuned per device.

```bash
torchrun --nproc_per_node=4 train_snp2p_model.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --train-bfile /path/to/train \
  --train-cov /path/to/train.cov --train-pheno /path/to/train.pheno \
  --val-bfile /path/to/val --val-cov /path/to/val.cov --val-pheno /path/to/val.pheno \
  --bt PHENOTYPE --batch-size 128 --jobs 8 \
  --sys2gene --sys2env --env2sys \
  --out outputs/run_ddp
```

 Frequently used options include `--snp2pheno` / `--gene2pheno` / `--sys2pheno` to control translation heads, `--mlm` for masked-SNP pretraining, and `--independent_predictors` for multi-phenotype outputs.

## Prediction and attention export

Use the trained checkpoint to generate predictions and (optionally) attention summaries. The loader reuses training metadata stored with the checkpoint.

```bash
python predict_attention.py \
  --onto samples/ontology.txt \
  --snp2gene samples/snp2gene.txt \
  --bfile /path/to/test \
  --cov /path/to/test.cov --pheno /path/to/test.pheno \
  --model outputs/run1/model_best.pth \
  --batch-size 256 --cuda 0 \
  --out outputs/run1/test
```

Outputs include:

- `{out}.prediction.csv`: predictions only (use `--prediction-only` to skip attention export).
- `{out}.attention.csv`: covariate, system, and gene attention values per individual.
- `{out}.sys_corr.csv`: correlation of system attention with predictions.
- `{out}.gene_corr.csv`: correlation of gene attention with predictions.

TSV inputs are supported via `--tsv` in place of `--bfile`.

## G2PT pipeline in overall

1. **Collapse Gene Ontology with your GWAS results**
   - Download GO Biological Process data with **[Prepare GO File](go_file.ipynb)** (thanks to @RiccardoIannaco).
   - Reduce the ontology to GWAS-relevant terms using **[Collapse Gene Ontology Based on Your GWAS Results](Collapse_Gene_Ontology_Based_on_GWAS_results.ipynb)** for a smaller, interpretable hierarchy.

2. **Train model**
   - Use the collapsed ontology and your genotype/covariate/phenotype files with `train_snp2p_model.py` (examples above or [train_model.sh](train_model.sh)).

3. **Predict with trained model**
   - Run `predict_attention.py` (see [predict_model.sh](predict_model.sh)) to obtain predictions and attention-derived importance scores.

4. **Analyze attention and epistasis**
   - Visualize high-importance systems: [Draw_ontology_with_highlighted_systems.ipynb](Draw_ontology_with_highlighted_systems.ipynb).
   - Plot attention flow: [Draw_Sankey.ipynb](Draw_Sankey.ipynb).
   - Search and visualize epistasis: [Epistasis_pipeline.ipynb](Epistasis_pipeline.ipynb).

## Future work

- [x] Applying [Differential Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer) to genetic factor translation.
- [x] Build data loader for `plink` binary file using [`sgkit`](https://sgkit-dev.github.io/sgkit/latest/).
- [x] Adding `.cov` and `.pheno` for input.
- [x] Change model for multiple phenotypes.
