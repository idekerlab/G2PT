# G2PT: A genotype-phenotype transformer to assess and explain polygenic risk

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

To train a new model using a data set, first make sure that you have
a proper virtual environment set up. Also make sure that you have all the required files
to run the training scripts:

1. Participant Genotype files:
    * You can put [PLINK binary file](https://www.cog-genomics.org/plink/1.9/input#bed) 
      * _--flip_ argument will flip ref. and alt. allele (`--flip` argument, which make homozygous ref. as 2)

2. Covariate and phenotype files
   * File including covariates and phenotypes.
   * same as `.cov` and `.pheno` in [PLINK](https://www.cog-genomics.org/plink/1.9/formats#cov)
     * If you want to use subset of covariates, you can put _--cov-ids_ (i.e. with `--cov-ids SEX AGE`, model will use only SEX and AGE as covaritates)
   * If you do not put `.cov` while you put PLINK bfiles. Covariates will be generated from `.fam` file (Sex only)  
   * If you do not put `.pheno`, you should include `PHENOTYPE` in training and validation covariate file 

* [Example of covariates file](samples/train.cov) (tab-separated)

| FID      | IID   | SEX | AGE | PC1 | PC2 | ... | PC10 |
|----------|-------|-----|-----|-----|-----| --- |------| 
| 10008090 | 10008090 |  1   | 48  | 3   | 0.3 | ... | 0.5  |

* [Example of phenotype file](samples/train.pheno) (tab-separated)

| FID      | IID   | PHENOTYPE |
|----------|-------|-----------| 
| 10008090 | 10008090 | 1.2       |

3. Ontology (hierarchy) file: 
    * _--onto_ : A tab-delimited file that contains the ontology (hierarchy) that defines the structure of a branch
    of a G2TP model that encodes the genotypes. The first column is always a term (subsystem or pathway),
    and the second column is a term or a gene.
    The third column should be set to "default" when the line represents a link between terms, (if you have nested subtree, you can put some name except 'gene').
    "gene" when the line represents an annotation link between a term and a gene.
    The following is an example describing a sample hierarchy.

        ![Ontology](./Figures/ontology.png)
      * _--subtree_order_ : if you have nested subtrees in ontology, you can set this option default is `['default']` (no subtree inside)

**If you want to collapse gene ontology based on a GWAS summary statistics please check first step of [G2PT overall pipeline](#1-collapse-gene-ontology-with-your-gwas-results)**

* [Example of ontology file](samples/ontology.txt) (header should not be included in file)

| parent     | child      | interaction_type |
|------------|------------|------------------|
| GO:0045834 | GO:0045923 | default          |
| GO:0045834 | GO:0043552 | default          |
| GO:0045923 | AKT2       | gene             |
| GO:0045923 | IL1B       | gene             |
| GO:0043552 | PIK3R4     | gene             |

  * _--snp2gene_ : A tab-delimited file for mapping SNPs to genes. The first column indicates SNP, second column for gene.
    * You can give additional information such as **chr**(chromosome), **pos**(position), **block_ind** (snp block information)
    * If you provide bfile as input, information from `.bim` will override additional information 

  
* [Example of snp2gene file](samples/snp2gene.txt)

| snp              | gene   | 
|------------------|--------|
| 16:56995236:A:C  | 	CETP	 |
| 8:126482077:G:A	 | TRIB1	 |
| 19:45416178:T:G	 | APOC1	 |
| 2:27752463:A:G	  | GCKR	  | 

* Example of snp2gene file with additional information

| snp              | gene   | chr | pos       |
|------------------|--------|-----|-----------|
| 16:56995236:A:C  | 	CETP	 | 16  | 56995236  |
| 8:126482077:G:A	 | TRIB1	 | 8   | 126482077 |
| 19:45416178:T:G	 | APOC1	 | 19  | 45416178  |
| 2:27752463:A:G	  | GCKR	  | 2   | 27752463  |


There are several optional parameters that you can provide in addition to the input files:

1. Propagation option:
   * _--sys2env_ : determines whether model will do Sys2Env propagation
   * _--env2sys_ : determines whether model will do Env2Sys propagation
   * _--sys2gene_ : determines whether model will do Gene2Sys propagation
2. Translation option:
   * _--sys2pheno_ : Updated system embeddings are used to predict phenotype
   * _--gene2pheno_ : Updated gene embeddings are used to predict phenotype
   * _--snp2pheno_ : SNP embeddings are used to predict phenotype
   * if you don't put any translation option, `sys2pheno` will be automatically set 
3. Model parameter:
   * _--hiddens-dims_: embedding and hierarchical transformer dimension size
4. Training parameters: 
   * _--epochs_ : the number of epoch to run during the training phase. The default is set to 256.
   * _--val-step_: Validation step
   * _--batch-size_ : the size of each batch to process at a time. The default is set to 256.
   * _--z-weight_ : for the continuous phenotype, individual with high absolute Z-score will be more sampled. if set as 0 (default), all population will be sampled in one training epoch   
   * _--dropout_: dropout option. Default is set 0.2
   * _--lr_ : Learning rate. Default is set 0.001.
   * _--wd_ : Weight decay. Default is set 0.001.
5. GPU option:
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
6. Model input and output:
   * _--model_: if you have trained model, put the path to the trained model.
   * _--out_: a name of directory where you want to store the trained models.

# G2PT Pipeline in Overall

## 1. Collapse Gene Ontology with your GWAS results

### Download Gene Ontology Data

Use the provided notebook to obtain Gene Ontology (Biological Process) data:

**[Prepare GO File](go_file.ipynb)**

*Special thanks to @RiccardoIannaco for this resource.*

### Collapse Gene Ontology

The complete Gene Ontology dataset is extensive, which can impact both interpretability and computational efficiency. To address this, we recommend collapsing the downloaded Gene Ontology based on your specific GWAS summary statistics.

Use the following notebook to streamline your Gene Ontology data:

**[Collapse Gene Ontology Based on Your GWAS Results](Collapse_Gene_Ontology_Based_on_GWAS_results.ipynb)**

This creates a focused and computationally manageable Gene Ontology subset tailored to your research needs.

## 2. Train model

You can put ontology file made from step 1.

### Model Training Example (Single GPU) 


**Warning: Sample data is randomly generated and won't produce meaningful results. Use sample data for tutorial purposes only and apply this pipeline to your actual data.**

Training script: [train_model.sh](train_model.sh) 

------------------


```          
python train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --train-bfile TRAIN --train-cov TRAIN.cov --train-pheno TRAIN.pheno \
                      --val-bfile VAL --train-cov VAL.cov --val-pheno VAL.pheno \
                      --epochs EPOCHS \
                      --lr LR \
                      --wd WD \
                      --batch_size BATCH_SIZE \
                      --dropout DROPOUT \
                      --val_step VAL_STEP \
                      --jobs JOBS \
                      --cuda 0 \
                      --hidden_dims HIDDEN_DIMS \
                      --out OUT \
                      --sys2env --env2sys --sys2gene \
                      --gene2pheno --sys2pheno \
                      --regression # if model is regression task
```

### Model Training Example (Multiple GPUs)

```          
python train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --train-bfile TRAIN --train-cov TRAIN.cov --train-pheno TRAIN.pheno \
                      --val-bfile VAL --train-cov VAL.cov --val-pheno VAL.pheno \
                      --epochs EPOCHS \
                      --lr LR \
                      --wd WD \
                      --batch_size BATCH_SIZE \
                      --dropout DROPOUT \
                      --val_step VAL_STEP \
                      --jobs JOBS \    
                      --dist-backend 'nccl' \
                      --dist-url 'tcp://127.0.0.1:2222' \ 
                      --multiprocessing-distributed \ 
                      --world-size 1 \ 
                      --rank 0 \
                      --hidden_dims HIDDEN_DIMS \
                      --out OUT \
                      --sys2env --env2sys --sys2gene \
                      --gene2pheno --sys2pheno \
                      --regression # if model is regression task
```

-----------------------------------
## 3. Predict with Trained Model

You can predict with a trained model.

please check [predict_model.sh](predict_model.sh)


```          
python predict_attention.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --bfile BFILE_prefix --cov COVAR.cov --pheno PHENO.pheno \
                      --model trained_model
                      --out output_prefix \ 
                      --batch_size BATCH_SIZE \
                      --jobs N_cpu 
```

This will generate

* Prediction: `{output_prefix}.prediction.csv`, containing only predictions (Good for performance evaluation!) 
* Attention result: `{output_prefix}.attention.csv`, containing Nx(C+S+G) covariates, system, and gene attention results for whole population
* System importance score: `{output_prefix}.sys_corr.csv`, containing correlation between system attention and prediction
* Gene importance score: `{output_prefix}.gene_corr.csv`, containing correlation between system attention and prediction

adding argument `--prediction-only` will make this script to predict only (no attention result)


## 4. Analyze Attention, Epistasis

## 4.1 Highlight systems in ontology

**This notebook is intended for real data analysis. Please follow these steps using your trained model.** 

Visualize ontology with highlighted high-importance systems (in tutorial I selected random systems): 

[Highlight systems in ontology](Draw_ontology_with_highlighted_systems.ipynb)

## 4.2 Draw Sankey Plot from Trained Model

**This notebook is intended for real data analysis. Please follow these steps using your trained model.**

Visualize attention flow from trained G2PT model:

[Draw Sankey from Model Attention](Draw_Sankey.ipynb)

## 4.3 Search Visualize, and Analyze Epistasis in system

**This notebook is intended for real data analysis. Please follow these steps using your trained model.**

Search for epistasis within systems and create visualizations:

[Epistais Search and Visualization Example](Epistasis_pipeline.ipynb)

## Future Works

- [x] Applying [Differential Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer) to genetic factor translation
- [x] Build data loader for `plink` binary file using [`sgkit`](https://sgkit-dev.github.io/sgkit/latest/) 
- [x] Adding `.cov` and `.pheno` for input
- [x] Change model for multiple phenotypes

