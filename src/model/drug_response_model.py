import torch
import torch.nn as nn

from src.model.utils import LayerNormNormedScaleOnly
from src.model.tree_conv import TreeConvolution
from src.model.compound import CompoundToSystems

class DrugResponseModel(nn.Module):

    def __init__(self, tree_parser, genotypes, compound_encoder, hidden_dims, dropout=0.2):
        super(DrugResponseModel, self).__init__()
        self.hidden_dims = hidden_dims
        self.tree_parser = tree_parser
        self.n_systems = self.tree_parser.n_systems
        self.n_genes = self.tree_parser.n_genes
        self.genotypes = genotypes

        print("Model is initialized with %d systems and %d gene mutations"%(self.n_systems, self.n_genes))
        print("Model will consider mutation types;", self.genotypes)
        '''
        #self.gene2gene_mask = torch.tensor(self.tree_parser.gene2gene_mask, dtype=torch.float32)
        
        self.gene_interaction_norm = nn.LayerNorm(hidden_dims)
        self.gene2gene = TreeConvolution(hidden_dims, int(hidden_dims/32), hidden_dims*4, self.gene_interaction_norm,
                                                dropout, conv_type='system')
        '''


        self.system_embedding = nn.Embedding(self.n_systems, hidden_dims)
        self.gene_embedding = nn.Embedding(self.n_genes, hidden_dims)

        self.mut_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.mut_update_norm_outer = nn.LayerNorm(hidden_dims)#LayerNormNormedScaleOnly(hidden_dims)

        self.sys_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.sys_update_norm_outer = nn.LayerNorm(hidden_dims)#LayerNormNormedScaleOnly(hidden_dims)

        self.gene_update_norm_inner = nn.LayerNorm(hidden_dims)
        self.gene_update_norm_outer = nn.LayerNorm(hidden_dims)

        self.prediction_norm_inner = nn.LayerNorm(hidden_dims)
        self.prediction_norm_outer = nn.LayerNorm(hidden_dims)



        self.sys_norm = nn.LayerNorm(hidden_dims)
        self.norm_channel_first = False

        self.mut2systems = nn.ModuleDict({genotype: TreeConvolution(hidden_dims, 4,
                                                                       hidden_dims*4, self.mut_update_norm_inner, self.mut_update_norm_outer, dropout,
                                                                    norm_channel_first=self.norm_channel_first, conv_type='genotype')
                                             for genotype in self.genotypes})

        self.system2system_forward = TreeConvolution(hidden_dims, 4, hidden_dims*4, self.sys_update_norm_inner, self.sys_update_norm_outer,
                                                dropout, norm_channel_first=self.norm_channel_first, conv_type='system')
        self.system2system_backward = TreeConvolution(hidden_dims, 4, hidden_dims * 4,
                                                     self.sys_update_norm_inner, self.sys_update_norm_outer,
                                                     dropout, norm_channel_first=self.norm_channel_first,
                                                     conv_type='system')

        self.system2gene = TreeConvolution(hidden_dims, 4, hidden_dims * 4,
                                                      self.gene_update_norm_inner, self.gene_update_norm_outer,
                                                      dropout, norm_channel_first=self.norm_channel_first,
                                                      conv_type='system')

        self.compound_encoder = compound_encoder
        self.compound_mapper_1 = nn.Linear(compound_encoder.hidden_layers[-1], hidden_dims)
        self.compound_norm = nn.LayerNorm(hidden_dims)
        self.compound_mapper_2 = nn.Linear(hidden_dims, hidden_dims)

        self.comp2sys_sys_norm = nn.LayerNorm(hidden_dims)
        self.comp2sys_comp_norm = nn.LayerNorm(hidden_dims)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

        self.comp2system = CompoundToSystems(hidden_dims, 1, hidden_dims*4, inner_norm=self.prediction_norm_inner,
                                             outer_norm=self.prediction_norm_outer,  dropout=dropout, transform=True)
        self.comp2gene = CompoundToSystems(hidden_dims, 1, hidden_dims*4, inner_norm=self.prediction_norm_inner,
                                             outer_norm=self.prediction_norm_outer, dropout=dropout, transform=True)

        self.drug_response_predictor = nn.Linear(hidden_dims*2, 1)


    def forward(self, genotype_dict, compound, nested_hierarchical_masks_forward, nested_hierarchical_masks_backward, sys2gene_mask, comp2sys_masks=None):
        batch_size = compound.size(0)

        system_embedding = self.system_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)
        gene_embedding = self.gene_embedding.weight.unsqueeze(0).expand(batch_size, -1, -1)

        system_embedding, mutation_effect = self.get_mut2system(system_embedding, gene_embedding, genotype_dict, )
        system_embedding, system_updates_forward = self.get_system2system(system_embedding, nested_hierarchical_masks_forward, direction='forward')
        system_embedding, system_updates_forward = self.get_system2system(system_embedding[-1][-1], nested_hierarchical_masks_backward, direction='backward')

        gene_embedding, system_effect = self.get_system2gene(system_embedding[-1][-1], gene_embedding, sys2gene_mask)
        compound_embedding = self.get_compound_embedding(compound, unsqueeze=True)

        system_embedding = system_embedding[-1][-1]

        compound_prediction = self.compound_encoder.predict(compound)
        compound_weighted_by_system = self.get_comp2system(compound_embedding, system_embedding, system_maks=comp2sys_masks)

        compound_weighted_by_genes = self.get_comp2gene(compound_embedding, gene_embedding, gene_mask=None)

        compound_attended = torch.cat([compound_weighted_by_system, compound_weighted_by_genes], dim=-1)
        #compound_attended = compound_weighted_by_system
        '''
        drug_response_prediction = self.dropout(
            self.activation(self.drug_response_batch_norm_1(self.drug_response_predictor_1(system_changed))))
        drug_response_prediction = self.dropout(
            self.activation(self.drug_response_batch_norm_2(self.drug_response_predictor_2(drug_response_prediction))))
        drug_response_prediction = self.drug_response_predictor_3(drug_response_prediction)
        '''
        drug_response_prediction = self.drug_response_predictor(compound_attended)

        return compound_prediction, drug_response_prediction

    def get_mut2system(self, system_embedding, mut_embedding, genotype_dict):
        #system_embedding = self.dropout(system_embedding/torch.norm(system_embedding, p=2, dim=-1, keepdim=True))
        #mut_embedding = self.dropout(mut_embedding/torch.norm(mut_embedding, p=2, dim=-1, keepdim=True))
        mutation_effects = {}
        for genotype in self.genotypes:
            mutation_effect = self.mut2systems[genotype].forward(system_embedding, mut_embedding, genotype_dict[genotype])
            mutation_effects[genotype] = mutation_effect
        for genotype in self.genotypes:
            system_embedding = system_embedding + mutation_effects[genotype]
        return system_embedding, mutation_effects

    def get_system2gene(self, system_embedding, gene_embedding, sys2gene_mask):
        system_effect = self.system2gene.forward(gene_embedding, system_embedding, sys2gene_mask)
        return gene_embedding + system_effect, system_effect

    def get_system2system(self, system_embedding, nested_hierarchical_masks, direction='forward'):
        #system_embedding = system_embedding
        updated_systems = []
        system_updates = []
        for hierarchical_masks in nested_hierarchical_masks:
            updated_systems_i = []
            system_update_i = []
            for hierarchical_mask in hierarchical_masks:
                if self.norm_channel_first:
                    system_embedding_input = system_embedding
                    system_embedding_input = system_embedding_input.transpose(-1, -2)
                    system_embedding_input = self.sys_norm(system_embedding_input)
                    system_embedding_input = self.dropout(system_embedding_input.transpose(-1, -2))
                else:
                    system_embedding_input = self.dropout(self.sys_norm(system_embedding))
                #system_embedding_input = system_embedding_input.transpose(-1, -2)
                #system_embedding_input = system_embedding_input.transpose(-1, -2)
                if direction=='forward':
                    tree_conv_result = self.system2system_forward(system_embedding_input, system_embedding_input, hierarchical_mask)
                else:
                    tree_conv_result = self.system2system_backward(system_embedding_input, system_embedding_input,
                                                                  hierarchical_mask)
                system_embedding = system_embedding + tree_conv_result
                updated_systems_i.append(system_embedding)
                system_update_i.append(tree_conv_result)
            updated_systems.append(updated_systems_i)
            system_updates.append(system_update_i)
        return updated_systems, system_updates


    def get_compound_embedding(self, compound, unsqueeze=True):
        compound_embedding = self.compound_encoder(compound)
        #compound_embedding = self.activation(self.compound_mapper_1(compound_embedding))
        compound_embedding = self.compound_mapper_1(compound_embedding)
        #compound_embedding = self.compound_mapper_2(self.compound_norm(self.activation(compound_embedding)))

        if unsqueeze:
            compound_embedding = compound_embedding.unsqueeze(1)

        return compound_embedding

    def get_comp2system(self, compound_embedding, system_embedding, system_maks=None, attention=False):
        #compound_embedding = self.comp2sys_comp_norm(compound_embedding)
        #system_embedding = self.comp2sys_sys_norm(system_embedding)
        #system_embedding = self.comp2sys_norm_1(self.activation(self.system_mapper_1(system_embedding)))
        #system_embedding = self.comp2sys_norm_2(self.system_mapper_2(system_embedding))

        comp2sys_result = self.comp2system(compound_embedding, system_embedding, system_embedding, mask=system_maks)
        if attention:
            comp2sys_attention = self.comp2system.get_attention(compound_embedding, system_embedding, system_embedding)
            return comp2sys_result, comp2sys_attention
        else:
            return comp2sys_result

    def get_comp2gene(self, compound_embedding, gene_embedding, attention=False, gene_mask=None):
        #compound_embedding = self.comp_norm_2(compound_embedding)
        #gene_embedding = self.comp2sys_norm_1(self.activation(self.system_mapper_1(gene_embedding)))
        #gene_embedding = self.comp2sys_norm_2(self.system_mapper_2(gene_embedding))
        comp2gene_result = self.comp2gene(compound_embedding, gene_embedding, gene_embedding, mask=gene_mask)
        if attention:
            comp2gene_attention = self.comp2gene.get_attention(compound_embedding, gene_embedding, gene_embedding)
            return comp2gene_result, comp2gene_attention
        else:
            return comp2gene_result













