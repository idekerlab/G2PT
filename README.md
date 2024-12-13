# G2PT: Mechanistic genotype-phenotype translation using hierarchical transformers

## Overview

Genome-wide association studies have linked millions of genetic variants to human phenotypes, but translating this information clinically has been challenged by a lack of biological understanding and widespread genetic interactions. 
With the advent of the Transformer deep learning architecture, new opportunities arise in creating predictive biological models that are both accurate and easily interpretable. 
Toward this goal we describe G2PT, a hierarchical Genotype-to-Phenotype Transformer that models bidirectional information flow among polymorphisms, genes, molecular systems, and phenotypes. 
G2PT effectively learns to predict metabolic traits in UK Biobank, including risk for diabetes and triglyceride-to-HDL cholesterol (TG/HDL) ratio, outperforming previous polygenic models. 


![Figure_1](./Figures/Figure_1.jpg)

## Environmental Set-Up

conda environment file environment.yml is provided
```
conda env create python==3.6 --name envname --file=environment.yml
```

## Usage

The usage of G2TP is similar to [DrugCell](https://github.com/idekerlab/DrugCell) and [NeST-VNN](https://github.com/idekerlab/nest_vnn)

To train a new model using a custom data set, first make sure that you have
a proper virtual environment set up. Also make sure that you have all the required files
to run the training scripts:

1. Participant Genotype files:
    * You can put [PLINK binary file](https://www.cog-genomics.org/plink/1.9/input#bed) 
    * Alternatively, you can use a tab-delimited file containing personal genotype data to reduce memory usage. 
      * Index will indicate Sample ID. 
      * `homozygous_a0`, `heterozygous`, `homozygous_a1` contain indices of SNP by the allele   

|         | homozygous_a0 | heterozygous | homozygous_a1 |
|---------|---------------|--------------|---------------|
| 1000909 | 0,1,3,5,7,9   | 2,4,5        | 6,8           |
| 1000303 | 1,3,6,7,8,9   | 2,5          | 4             |

2. Ontology (hierarchy) file: _--onto_ :
    * A tab-delimited file that contains the ontology (hierarchy) that defines the structure of a branch
    of a G2TP model that encodes the genotypes. The first column is always a term (subsystem or pathway),
    and the second column is a term or a gene.
    The third column should be set to "default" when the line represents a link between terms, (if you have nested subtree, you can put some name except 'gene').
    "gene" when the line represents an annotation link between a term and a gene.
    The following is an example describing a sample hierarchy.

        ![Ontology](./Figures/ontology.png)

| parent     | child      | interaction_type |
|------------|------------|------------------|
| GO:0045834 | GO:0045923 | default          |
| GO:0045834 | GO:0043552 | default          |
| GO:0045923 | AKT2       | gene             |
| GO:0045923 | IL1B       | gene             |
| GO:0043552 | PIK3R4     | gene             |
  * _--snp2gene_ : A tab-delimited file for mapping SNPs to genes. The first column indicates SNP, second column for gene, and third for chromosome

| SNP_ID           | Gene       | Chromosome |
|------------------|------------|------------|
| 16:56995236:A:C  |	CETP	| 16 |
|8:126482077:G:A	| TRIB1	| 8 |
|19:45416178:T:G	| APOC1	| 19 |
|2:27752463:A:G	| GCKR	| 2 |

  * _--subtree_order_ : if you have nested subtrees in ontology, you can set this option default is `['default']` (no subtree inside)


3. Training, validation, test covariates files
   * Training, validation, test file include FID, IID, phenotype value, and covariates.
   * same as `.cov` in [PLINK](https://www.cog-genomics.org/plink/1.9/formats#cov)
     * If you do not put `.cov` while you put PLINK bfiles, covariates will be generated from `.fam` file.  
   * You should include `PHENOTYPE` in training and validation covariate file 

| FID      | IID   | PHENOTYPE | SEX | AGE | COV1 | COV2 | ... | COVN |
|----------|-------|-----------|-----|-----|------|--------| --- |--------| 
| 10008090 | 10008090 | 1.2       | 1   | 48  | 3    | 0.3    | ... | 0.5    |


There are several optional parameters that you can provide in addition to the input files:

1. Propagation option:
   * _--sys2env_ : determines whether model will do Sys2Env propagation
   * _--env2sys_ : determines whether model will do Env2Sys propagation
   * _--sys2gene_ : determines whether model will do Gene2Sys propagation
2. Model parameter:
   * _--hiddens-dims_: embedding and hierarchical transformer dimension size
3. Training parameters: 
   * _--epochs_ : the number of epoch to run during the training phase. The default is set to 256.
   * _--val-step_: Validation step to measure performance on validation dataset during training
   * _--batch-size_ : the size of each batch to process at a time. The default is set to 5000.
You may increase this number to speed up the training process within the memory capacity
   * _--z-weight_ : for the continuous phenotype, with high `z-weight` will be more sampled  
   * _--dropout_: dropout option. Default is set 0.2
   * _--lr_ : Learning rate. Default is set 0.001.
   * _--wd_ : Weight decay. Default is set 0.001.
4. GPU option:
   * Single GPU option
     * _--cuda_ : the ID of GPU unit that you want to use for the model training. The default setting
     is to use GPU 0.
   * Multi GPU option (multi-node will be supported)
     * _--multiprocessing-distributed_ : determines whether model will be trained in multi-gpu distributed set-up
     * _--world-size_ : size of world, default is 1
     * _--rank_ : rank, default is 0
     * _--local-rank_ : local rank, default is 0
     * _--dist-url_ : distribute url, `tcp://127.0.0.1:2222`
     * _--dist_backend_ : distribute backend default is `nccl`
5. Model input and output:
   * _--model_: if you have trained model to load, put the path to the trained model.
   * _--out_: a name of directory where you want to store the trained models.


### Model Training Example (Single GPU) 

```          
usage: train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --train-bfile train  --train-cov train.cov \
                      --val-bfile val  --val-cov val.cov \
                      --test-bfile train  --test-cov test.cov \
                      --epochs EPOCHS \
                      --lr LR \
                      --wd WD \
                      --batch-size BATCH_SIZE \
                      --dropout DROPOUT \
                      --val-step VAL_STEP \
                      --jobs JOBS \
                      --cuda 0 \
                      --hidden-dims HIDDEN_DIMS \
                      --out OUT
```

### Model Training Example (Multiple GPUs)

```          
usage: train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --train-bfile train  --train-cov train.cov \
                      --val-bfile val  --val-cov val.cov \
                      --test-bfile train  --test-cov test.cov \
                      --epochs EPOCHS \
                      --lr LR \
                      --wd WD \
                      --batch_size BATCH_SIZE \
                      --dropout DROPOUT \
                      --val-step VAL-STEP \
                      --jobs JOBS \    
                      --dist-backend 'nccl' \
                      --dist-url 'tcp://127.0.0.1:2222' \ 
                      --multiprocessing-distributed \ 
                      --world-size 1 \ 
                      --rank 0 \
                      --hidden-dims HIDDEN_DIMS \
                      --out OUT
```


## Predict Phenotypes and Get Attention Values from a Cohort

We provide code to predict phenotypes for a cohort and retrieve attention values for them.

```shell
python \
      predict_attention.py \
  --onto ONTO \
  --snp2gene SNP2Gene \
  --cpu 8 --cuda 0 --batch 8 \
  --model G2PT_output \
  --bfile cohort_bfile \
  --cov cohort_cov   \
  --out output_dir \
  --system_annot sys_annot_file
```

`--system_annot` is an optional argument. Its first column should contain system IDs, and the second column should contain the corresponding descriptions of the systems.

This script generates the following outputs in CSV format:

* Attention Matrix:
  * Size: `n_samples × (n_functions + n_genes)`
  * File: `output_dir.attention.csv`
* System Importance Scores:
  * File: `output_dir.sys_corr.csv`
* Gene Importance Scores:
  * File: `output_dir.gene_corr.csv`


## Future Works

- [x] Applying [Differential Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer) to genetic factor translation
- [x] Build data loader for `plink` binary file using [`sgkit`](https://sgkit-dev.github.io/sgkit/latest/) 
- [x] Inferring covariates from `.fam`
