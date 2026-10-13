import torch.nn as nn
import torch
from src.model.hierarchical_transformer.attention import FewShotAttention



class DrugResponseFewShotTransformer(nn.Module):

    def __init__(self, hidden_dims, n_train_celllines, dropout=0.5):
        super(DrugResponseFewShotTransformer, self).__init__()
        self.hidden_dims = hidden_dims

        self.few_shot_attention = FewShotAttention(self.hidden_dims, n_heads=1, n_train_celllines=n_train_celllines, dropout=dropout)
        #self.few_shot_attention_gene = FewShotAttention(self.hidden_dims, self.n_heads, dropout=dropout)
        self.predictor = nn.Linear(hidden_dims * 2, 1)


    def forward(self, query_sys_embedding, query_gene_embedding, key_sys_embedding, key_gene_embedding, transform=True):
        batch_size, n_sys, hidden_dim = query_sys_embedding.size()
        batch_size, n_gene, hidden_dim = query_gene_embedding.size()

        query_sys_embedding = query_sys_embedding.unsqueeze(1) # batch_size, 1, n_sys, hidden_dims
        query_gene_embedding = query_gene_embedding.unsqueeze(1)  # batch_size, 1, n_gene, hidden_dims

        key_sys_embedding = key_sys_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1) # batch_size, n_train_cellline, n_sys, hidden_dims
        key_gene_embedding = key_gene_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)  # batch_size, n_train_cellline, n_gene, hidden_dims

        sys_transformed = []
        for i in range(n_sys):
            embedding_result, attention, score = self.few_shot_attention(query_sys_embedding[:, :, i, :], key_sys_embedding[:, :, i, :], key_sys_embedding[:, :, i, :], transform=transform)
            sys_transformed.append(embedding_result)

        gene_transformed = []
        for j in range(n_gene):
            embedding_result, attention, score = self.few_shot_attention(query_gene_embedding[:, :, j, :], key_gene_embedding[:, :, j, :], key_gene_embedding[:, :, j, :], transform=transform)
            gene_transformed.append(embedding_result)

        sys_transformed = torch.stack(sys_transformed, dim=2)[:, 0, :, :]
        gene_transformed = torch.stack(gene_transformed, dim=2)[:, 0, :, :]

        return sys_transformed, gene_transformed