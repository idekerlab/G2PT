python train_snp2p_model.py \
    --onto ./samples/ontology.txt \
    --snp2gene ./samples/snp2gene.txt \
    --train-bfile ./samples/train \
    --train-cov ./samples/train.cov \
    --train-pheno ./samples/train.pheno \
    --val-bfile ./samples/val \
    --val-cov ./samples/val.cov \
    --val-pheno ./samples/val.pheno \
    --jobs 8 \
    --epochs 21 \
    --hidden-dims 64 \
    --lr 0.0001 \
    --wd 0.0001 \
    --subtree-order default \
    --dropout 0.2 \
    --batch-size 128 \
    --val-step 5 \
    --z-weight 0 \
    --out ./samples/output_model.pt \
    --cuda 0 \
    --sys2env \
    --env2sys \
    --sys2gene \
    --gene2pheno \
    --sys2pheno \
    --regression

