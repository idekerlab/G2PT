# Epistasis Simulation

## Overview

The epistasis simulation utilities generate synthetic genotype/phenotype datasets
with both additive and pairwise interaction effects. The simulation is designed
for benchmarking discovery workflows and includes options for LD structure,
heritability budgets, and biasing epistatic pairs toward non-significant SNPs.
A helper function can also build a hierarchical SNP→Gene→System ontology that
is aligned with the simulated causal structure.

For a full walkthrough, see the notebook: [`Epistasis_simulation.ipynb`](../../Epistasis_simulation.ipynb).

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

## Outputs

- `simulate_epistasis` returns a dictionary with the genotype matrix, phenotype,
  causal SNP indices, epistatic pairs, effect sizes, and LD block assignments.
- `build_hierarchical_ontology` returns three DataFrames suitable for use with
  tree utilities and downstream analyses: SNP-to-gene, gene-to-system, and the
  system hierarchy.
