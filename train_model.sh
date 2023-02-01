python train_model.py \
        --onto ./sample/ontology.txt \
	--gene2id ./sample/gene2ind.txt \
	--cell2id ./sample/cell2ind.txt \
	--genotypes mutation:./sample/cell2mutation.txt,cna:./sample/cell2cnamplification.txt,cnd:./sample/cell2cndeletion.txt \
	--jobs 16 --cuda 0 --compound_epochs 0 --epochs 1501 --hidden_dims 128 --lr 0.0001 --wd 0.01 --subtree_order default --compound_layers 512 128 --dropout 0.2 --l2_lambda 0.001 --batch_size 32 --z_weight 2 --radius 2 --n_bits 2048  --val_step 10 \
	--out ./sample/output_model.pt \
	--train ./sample/training_data.txt \
	--val ./sample/test_data.txt 

