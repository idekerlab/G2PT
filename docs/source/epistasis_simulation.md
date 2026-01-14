# Epistasis Simulation

## Overview

The epistasis simulation utilities generate synthetic genotype/phenotype datasets
with both additive and pairwise interaction effects. The simulation is designed
for benchmarking discovery workflows and includes options for LD structure,
heritability budgets, and biasing epistatic pairs toward non-significant SNPs.
A helper function can also build a hierarchical SNP→Gene→System ontology that
is aligned with the simulated causal structure.

For a full walkthrough, see the notebook: [`Epistasis_simulation.ipynb`](../../Epistasis_simulation.ipynb).

## Key features

- **Additive + epistatic signal control**: Allocate separate heritability budgets
  for additive and pairwise interaction effects.
- **LD-aware genotypes**: Optionally simulate LD blocks and correlated SNPs.
- **Ontology-aligned mappings**: Generate SNP→Gene and Gene→System mappings that
  align with the simulated causal pairs, with tunable system coherence.
- **Export-ready artifacts**: Write out `genotypes.tsv`, `simulation.pheno`,
  `simulation.cov`, `ontology.tsv`, and `snp2gene.tsv` for downstream pipelines.

## Workflow overview

1. Configure simulation hyperparameters.
2. Simulate genotypes (`G`) and phenotype (`y`).
3. Build an ontology and mapping files aligned to the simulation.
4. Export data, ontology, and causal metadata to disk.
5. Train a model, predict attention, and evaluate epistasis retrieval.

## Usage

### Configure the simulation

```python
output_dir = "./epistasis_simulation_samples/"
seed = 42
n_samples = 10000
n_pairs = 50
h2_epistatic = 0.1
ontology_coherence = 0.5
n_additive = 100
```

### Simulate genotypes and phenotypes

The main simulation returns a dictionary with genotype matrix `G`, phenotype `y`,
and causal SNP indices and pairs.

```python
from src.utils.analysis.epistasis_simulation import simulate_epistasis

sim = simulate_epistasis(
    n_samples=2000,
    n_snps=50000,
    n_additive=200,
    n_pairs=75,
    h2_additive=0.4,
    h2_epistatic=0.2,
    n_ld_blocks=200,
    ld_rho=0.8,
)

print(sim["G"].shape)      # (samples, snps)
print(len(sim["pair_idx"]))
```

### Build a hierarchical ontology aligned to the simulation

`build_hierarchical_ontology` maps simulated SNPs to genes and systems and
builds a multi-level system hierarchy. The `ontology_coherence` parameter
controls whether epistatic pairs are placed in the same, related, or distant
systems.

```python
from src.utils.analysis.epistasis_simulation import build_hierarchical_ontology

snp_df, gene_df, system_df = build_hierarchical_ontology(
    sim,
    n_genes=800,
    n_systems=60,
    ontology_coherence=0.5,
    overlap_prob=0.25,
    n_causal_systems=20,
)

snp_df.to_csv("snp2gene.tsv", sep="\t")
gene_df.to_csv("gene2system.tsv", sep="\t", index=False)
system_df.to_csv("system_hierarchy.tsv", sep="\t", index=False)
```

### Export genotypes, phenotypes, and covariates

The model training and evaluation utilities expect TSV files for genotypes,
phenotypes, and covariates. The notebook uses IID/FID pairs to keep identifiers
consistent across files.

```python
import pandas as pd
import numpy as np

iids = [f"sample_{i}" for i in range(sim["y"].shape[0])]

genotypes_df = pd.DataFrame(sim["G"], columns=snp_df.index.values)
genotypes_df.index = iids
genotypes_df.index.name = "IID"
genotypes_df.to_csv(f"{output_dir}/genotypes.tsv", sep="\t")

pheno_df = pd.DataFrame({"FID": iids, "IID": iids, "phenotype": sim["y"]})
pheno_df.to_csv(f"{output_dir}/simulation.pheno", index=False, sep="\t")

cov_df = pd.DataFrame(
    {
        "FID": iids,
        "IID": iids,
        "SEX": np.random.randint(2, size=sim["y"].shape[0]),
        "AGE": np.random.randint(40, 70, size=sim["y"].shape[0]),
    }
)
cov_df.to_csv(f"{output_dir}/simulation.cov", index=False, sep="\t")
```

### Export ontology and causal metadata

The ontology file combines gene-to-system and system-to-supersystem edges into
a single `parent`, `child`, `interaction` format expected by the training and
analysis pipelines. The causal metadata file is used later for evaluation.

```python
import json

gene_df = gene_df.rename(columns={"gene_id": "child", "system_id": "parent"})
gene_df["interaction"] = "gene"
system_df = system_df.rename(columns={"system_id": "child", "supersystem_id": "parent"})
system_df["interaction"] = "default"

ontology_df = pd.concat(
    [
        gene_df[["parent", "child", "interaction"]],
        system_df[["parent", "child", "interaction"]],
    ]
)
ontology_df.to_csv(f"{output_dir}/ontology.tsv", index=False, sep="\t", header=False)
snp_df.to_csv(f"{output_dir}/snp2gene.tsv", index=True, sep="\t")

causal_info = {
    "epistatic_pairs": [list(map(int, p)) for p in sim["pair_idx"]],
    "additive_snps": list(map(int, sim["additive_idx"])),
}
with open(f"{output_dir}/causal_info.json", "w") as f:
    json.dump(causal_info, f, indent=2)
```

### Train, predict attention, and evaluate retrieval

Once the artifacts are saved, you can train a model, compute attention scores,
and run epistasis retrieval evaluation as in the notebook.

```bash
python train_snp2p_model.py \
  --train-tsv "epistasis_simulation_samples/genotypes.tsv" \
  --train-pheno "epistasis_simulation_samples/simulation.pheno" \
  --train-cov "epistasis_simulation_samples/simulation.cov" \
  --onto "epistasis_simulation_samples/ontology.tsv" \
  --snp2gene "epistasis_simulation_samples/snp2gene.tsv" \
  --out "epistasis_simulation_samples/output_model.txt" \
  --epochs 51 \
  --batch-size 64 \
  --lr 1e-4 \
  --qt "phenotype" \
  --jobs 4 \
  --cuda 0 \
  --sys2env --env2sys --sys2gene \
  --sys2pheno --gene2pheno \
  --val-step 50 \
  --use_hierarchical_transformer

python predict_attention.py \
  --model "epistasis_simulation_samples/output_model.txt.50" \
  --tsv "epistasis_simulation_samples/genotypes.tsv" \
  --pheno "epistasis_simulation_samples/simulation.pheno" \
  --cov "epistasis_simulation_samples/simulation.cov" \
  --onto "epistasis_simulation_samples/ontology.tsv" \
  --snp2gene "epistasis_simulation_samples/snp2gene.tsv" \
  --out "epistasis_simulation_samples/output_model.txt.50" \
  --batch-size 256 \
  --cuda 0
```

```python
from src.utils.analysis.epistasis_retrieval_evaluation import (
    EvaluationConfig,
    EpistasisRetrievalEvaluator,
)

config = EvaluationConfig(
    causal_info="epistasis_simulation_samples/causal_info.json",
    attention_results="epistasis_simulation_samples/output_model.txt.50.phenotype.head_sum.csv",
    system_importance="epistasis_simulation_samples/output_model.txt.50.phenotype.head_sum.sys_importance.csv",
    tsv="epistasis_simulation_samples/genotypes.tsv",
    pheno="epistasis_simulation_samples/simulation.pheno",
    cov="epistasis_simulation_samples/simulation.cov",
    onto="epistasis_simulation_samples/ontology.tsv",
    snp2gene="epistasis_simulation_samples/snp2gene.tsv",
    top_n_systems=5,
    snp_threshold=50,
    num_workers=1,
    executor_type="threads",
    quantiles=[0.9],
    output_prefix="epistasis_simulation_samples/simulation_output.50",
)

evaluator = EpistasisRetrievalEvaluator(config)
evaluator.evaluate()
```

### Optional: statistical sanity checks

The notebook includes quick checks to verify that additive and interaction
effects are detectable in the synthetic dataset. For example, you can fit a
linear model per SNP for additive signals and evaluate interaction terms for
known epistatic pairs.

```python
import statsmodels.api as sm

results = []
for snp in genotypes.columns:
    df_snp = df_full[["phenotype", "SEX", "AGE", snp]].dropna()
    X = sm.add_constant(df_snp[["SEX", "AGE", snp]])
    model = sm.OLS(df_snp["phenotype"], X).fit()
    results.append((snp, model.pvalues[snp]))
```

## Outputs

- `simulate_epistasis` returns a dictionary with the genotype matrix, phenotype,
  causal SNP indices, epistatic pairs, effect sizes, and LD block assignments.
- `build_hierarchical_ontology` returns three DataFrames suitable for use with
  tree utilities and downstream analyses: SNP-to-gene, gene-to-system, and the
  system hierarchy.
- Exported artifacts on disk typically include:
  - `genotypes.tsv`: Genotype dosages with IID index.
  - `simulation.pheno`: Phenotype values with FID/IID.
  - `simulation.cov`: Covariates (e.g., SEX, AGE) with FID/IID.
  - `ontology.tsv`: Combined system and gene edges for the parser.
  - `snp2gene.tsv`: SNP-to-gene mapping.
  - `causal_info.json`: Ground-truth additive SNPs and epistatic pairs.
