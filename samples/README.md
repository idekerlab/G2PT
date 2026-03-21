# Notice on Synthetic Sample Data

All files in the `samples/` directory contain fully synthetic data generated solely for demonstration and testing purposes.

These sample files do **not** contain real participant-level data, real genotypes, real phenotypes, or real derived genetic data.

Any identifiers included in the sample files are provided only as placeholders for formatting or software compatibility and should not be interpreted as evidence that the files contain real participant data.

The repository is intended to share code only. No real participant data should be distributed through this repository.

---

## Regenerating the Sample Files

Use `generate_synthetic_data.py` to reproduce or customise the sample files.

### Requirements

The script depends on packages already present in the `G2PT_github` conda environment:
`numpy`, `scipy`, `pandas`.

### Basic usage (reproduces the default samples)

```bash
/cellar/users/i5lee/miniconda3/envs/G2PT_github/bin/python samples/generate_synthetic_data.py
```

This writes all output files into the `samples/` directory by default.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--seed` | `42` | Random seed for reproducibility |
| `--n_train` | `1000` | Number of training samples |
| `--n_val` | `500` | Number of validation samples |
| `--n_snps` | `293` | Number of SNPs |
| `--n_genes` | `150` | Number of unique genes |
| `--n_go_terms` | `80` | Number of GO-like ontology terms |
| `--out_dir` | *(script dir)* | Output directory |
| `--prefix_train` | `synthetic_train` | File prefix for the train split |
| `--prefix_val` | `synthetic_val` | File prefix for the val split |

### Example: larger custom dataset

```bash
/cellar/users/i5lee/miniconda3/envs/G2PT_github/bin/python samples/generate_synthetic_data.py \
    --n_train 5000 --n_val 1000 --n_snps 1000 \
    --n_genes 400 --n_go_terms 200 \
    --out_dir /path/to/output --seed 99
```

### Output files

| File | Format | Description |
|------|--------|-------------|
| `*.bed` | PLINK BED (binary, SNP-major) | Genotype dosages 0/1/2 |
| `*.bim` | PLINK BIM (tab-separated) | SNP metadata: `CHR SNP_ID CM POS A1 A2` |
| `*.fam` | PLINK FAM (space-separated) | Sample metadata: `FID IID PID MID SEX PHENO` |
| `*.pheno` | Tab-separated, with header | Continuous phenotype: `FID IID PHENOTYPE` |
| `*.cov` | Tab-separated, with header | Covariates: `FID IID SEX AGE AGE2 PC1…PC10` |
| `snp2gene.txt` | Tab-separated, with header | SNP-to-gene mapping: `snp gene` |
| `ontology.txt` | Tab-separated, no header | GO-like DAG edges: `parent child type` (`type` = `default` or `gene`) |

### How data are generated

1. **Genotypes** are simulated with a realistic MAF spectrum and LD block structure via `simulate_epistasis()` (see `src/utils/analysis/epistasis_simulation.py`).
2. **Phenotype** combines additive SNP effects, pairwise epistatic interactions, and Gaussian noise, scaled to unit variance.
3. **SNP IDs** follow the `CHR:POS:REF:ALT` convention used by UK Biobank / PLINK, with positions drawn from GRCh38 chromosome lengths.
4. **Covariates** include binary sex, age (uniform 40–80), age², and 10 simulated principal components.
5. **Ontology** is a GO-like DAG: leaf terms connect to genes (`gene` edges); intermediate terms connect upward to a single root (`default` edges).