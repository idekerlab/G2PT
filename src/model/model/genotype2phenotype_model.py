import torch
import torch.nn as nn

from src.model.model import Genotype2PhenotypeTransformer
from src.model.hierarchical_transformer import Genotype2Phenotype


class Genotype2PhenotypeModel(Genotype2PhenotypeTransformer):

    def __init__(self, tree_parser, genotypes, hidden_dims, dropout=0.2):
        super(Genotype2PhenotypeModel, self).__init__(tree_parser, genotypes, hidden_dims, dropout=dropout)

        #phenotype_vector = torch.empty(1, hidden_dims)
        #nn.init.xavier_normal(phenotype_vector)
        self.phenotype_vector = nn.Embedding(1, hidden_dims)#nn.Parameter(phenotype_vector, requires_grad=True)
        #self.phenotype_mapper_1 = nn.Linear(hidden_dims, hidden_dims)
        #self.phenotype_norm = nn.LayerNorm(hidden_dims)
        #self.phenotype_mapper_2 = nn.Linear(hidden_dims, hidden_dims)

        self.sys2pheno_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys2pheno_norm_outer = nn.LayerNorm(hidden_dims)
        self.gene2pheno_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene2pheno_norm_outer = nn.LayerNorm(hidden_dims)

        self.system2phenotype = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.sys2pheno_norm_inner,
                                                   outer_norm=self.sys2pheno_norm_outer, dropout=dropout, transform=True)
        self.gene2phenotype = Genotype2Phenotype(hidden_dims, 1, hidden_dims * 4, inner_norm=self.gene2pheno_norm_inner,
                                                 outer_norm=self.gene2pheno_norm_outer, dropout=dropout, transform=True)

        self.phenotype_predictors = nn.ModuleList([nn.Linear(hidden_dims*2, 1) for i in range(3)])

    def forward(self, genotype_dict, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward,
                sys2gene_mask, gene_weight=None, sys2cell=True, cell2sys=True, sys2gene=True):

        system_embedding_result, gene_embedding_result = super(Genotype2PhenotypeModel, self).forward(genotype_dict,
                                                                                                      nested_hierarchical_masks_forward,
                                                                                                      nested_hierarchical_masks_backward,
                                                                                                      sys2gene_mask=sys2gene_mask,
                                                                                                      gene_weight=gene_weight,
                                                                                                      sys2cell=sys2cell,
                                                                                                      cell2sys=cell2sys,
                                                                                                      sys2gene=sys2gene)
        batch_size = system_embedding_result[0].size(0)
        phenotype_vector = self.get_phenotype_vector(batch_size)

        predictions = [self.prediction(predictor, phenotype_vector, system_embedding, gene_embedding) for
                       predictor, system_embedding, gene_embedding in
                       zip(self.phenotype_predictors, system_embedding_result, gene_embedding_result)]

        return predictions

    def get_phenotype_vector(self, batch_size=1):
        phenotype_vector = self.phenotype_vector.weight.unsqueeze(0).expand(batch_size, -1, -1)
        #phenotype_vector = self.activation(self.phenotype_mapper_1(phenotype_vector))
        # compound_embedding = self.compound_mapper_1(compound_embedding)
        #phenotype_vector = self.phenotype_mapper_2(self.phenotype_norm(phenotype_vector))
        return phenotype_vector

    def prediction(self, predictor, phenotype_vector, system_embedding, gene_embedding):
        phenotype_weighted_by_systems = self.get_system2phenotype(phenotype_vector, system_embedding, system_mask=None)
        phenotype_weighted_by_genes = self.get_gene2phenotype(phenotype_vector, gene_embedding, gene_mask=None)
        phenotype_attended = torch.cat([phenotype_weighted_by_systems, phenotype_weighted_by_genes], dim=-1)
        phenotype_prediction = predictor(phenotype_attended)
        return phenotype_prediction

    def get_system2phenotype(self, phenotype_vector, system_embedding, system_mask=None, attention=False, score=False):
        sys2phenotype_result = self.system2phenotype.forward(phenotype_vector, system_embedding, system_embedding, mask=system_mask)
        if attention:
            sys2phenotype_attention = self.system2phenotype.get_attention(phenotype_vector, system_embedding, system_embedding)
            sys2phenotype_result = [sys2phenotype_result, sys2phenotype_attention]
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding, system_embedding)
                sys2phenotype_result += [sys2phenotype_score]
            return sys2phenotype_result
        else:
            if score:
                sys2phenotype_score = self.system2phenotype.get_score(phenotype_vector, system_embedding,
                                                                      system_embedding)
                sys2phenotype_result = [sys2phenotype_result, sys2phenotype_score]
            return sys2phenotype_result


    def get_gene2phenotype(self, phenotype_vector, gene_embedding, gene_mask=None, attention=False, score=False):
        gene2phenotype_result = self.gene2phenotype.forward(phenotype_vector, gene_embedding, gene_embedding, mask=gene_mask)
        if attention:
            gene2phenotype_attention = self.gene2phenotype.get_attention(phenotype_vector, gene_embedding, gene_embedding)
            gene2phenotype_result = [gene2phenotype_result, gene2phenotype_attention]
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding, gene_embedding)
                gene2phenotype_result += [gene2phenotype_score]
            return gene2phenotype_result
        else:
            if score:
                gene2phenotype_score = self.gene2phenotype.get_score(phenotype_vector, gene_embedding,
                                                                      gene_embedding)
                gene2phenotype_result = [gene2phenotype_result, gene2phenotype_score]
            return gene2phenotype_result