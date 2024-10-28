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

1. Genotype indexing files:
    * Tab-delimited files to index snp, gene. 
      * _--snp2id_ : index of SNP, the first column indicates index and the second column indicates SNP ID 
      * _--gene2id.txt_ : index of gene, the first column indicates index and the second column indicates gene
2. Participant Genotype files:
    * Tab-delimited file containing personal genotype data. 
      * Index will indicate Sample ID. 
      * `homozygous_a0`, `heterozygous`, `homozygous_a1` contain index of SNP by the allele   

|         | homozygous_a0 | heterozygous | homozygous_a1 |
|---------|---------------|--------------|---------------|
| 1000909 | 0,1,3,5,7,9   | 2,4,5        | 6,8           |
| 1000303 | 1,3,6,7,8,9   | 2,5          | 4             |

4. Ontology (hierarchy) file: _--onto_ :
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


5. Training, validation, test files
   * Training, validation, test file include sample ID, response value, and covariates. 

| Sample_ID | response | sex | age | age_sq | cov_1 | cov_2 | ... | cov_n |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | 
| 10008090 | 1.343 | 1 | 64 | 3489 | 0.2 | 0.3 | ... | 0.5 |


There are several optional parameters that you can provide in addition to the input files:

1. Propagation option:
   * _--sys2env_ : determines whether model will do Sys2Env propagation
   * _--env2sys_ : determines whether model will do Env2Sys propagation
   * _--sys2gene_ : determines whether model will do Gene2Sys propagation
2. Model parameter:
   * _--hiddens_dims_: embedding and hierarchical transformer dimension size
3. Training parameters: 
   * _--epochs_ : the number of epoch to run during the training phase. The default is set to 256.
   * _--val_step_: Validation step
   * _--batch_size_ : the size of each batch to process at a time. The default is set to 5000.
You may increase this number to speed up the training process within the memory capacity
   * _--z_weight_ : for the continuous phenotype, with high `z_weight` will be more sampled  
   * _--dropout_: dropout option. Default is set 0.2
   * _--lr_ : Learning rate. Default is set 0.001.
   * _--wd_ : Weight decay. Default is set 0.001.
4. GPU option:
   * Single GPU option
     * _--cuda_ : the ID of GPU unit that you want to use for the model training. The default setting
     is to use GPU 0.
   * Multi GPU option (multi-node will be supported)
     * _--multiprocessing-distributed_ : determines whether model will be trained in multi-gpu distributed set-up
     * _--world_size_ : size of world, default is 1
     * _--rank_ : rank, default is 0
     * _--local_rank_ : local rank, default is 0
     * _--dist_url_ : distribute url, `tcp://127.0.0.1:2222`
     * _--dist_backend_ : distribute backend default is `nccl`
5. Model input and output:
   * _--model_: if you have trained model, put the path to the trained model.
   * _--out_: a name of directory where you want to store the trained models.


### Model Training Example (Single GPU) 

```          
usage: train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --snp2id SNP2ID \
                      --gene2id GENE2ID \ 
                      --genotype genotype_file_dir \
                      --train TRAIN --val VAL --test TEST \
                      --epochs EPOCHS \
                      --lr LR \
                      --wd WD \
                      --batch_size BATCH_SIZE \
                      --dropout DROPOUT \
                      --val_step VAL_STEP \
                      --jobs JOBS \
                      --cuda 0 \
                      --hidden_dims HIDDEN_DIMS \
                      --out OUT
```

### Model Training Example (Multiple GPUs)

```          
usage: train_snp2p_model.py \
                      --onto ONTO \
                      --snp2gene SNP2Gene \
                      --snp2id SNP2ID \
                      --gene2id GENE2ID \ 
                      --genotype genotype_file_dir \
                      --train TRAIN --val VAL --test TEST \
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
                      --out OUT
```


you can train model with sample data by using [train_model.sh](train_model.sh)


## Future Works

- [x] Applying [Differential Transformer](https://github.com/microsoft/unilm/tree/master/Diff-Transformer) to genetic factor translation
- [ ] Build data loader for `plink` binary file using [`pandas-plink`](https://pypi.org/project/pandas-plink/) 

