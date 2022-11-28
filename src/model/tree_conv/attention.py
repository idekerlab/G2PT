import torch.nn as nn
import torch.nn.functional as F
import torch



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, activation='softmax', top_k=0):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(activation=activation, top_k=top_k)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.dropout(self.output_linear(self.dropout(x)))

    def get_attention(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return attn


class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, activation='softmax', top_k=0):
        super().__init__()
        if activation=='softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation=='sigmoid':
            self.activation = nn.Sigmoid()
        elif activation=='tanh':
            self.activation = nn.Tanh()
        #self.top_k = top_k

    def forward(self, query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / torch.sqrt(torch.tensor(query.size(-1)))
        if mask is not None:
            mask = mask == 0
            scores = scores.masked_fill(mask, -1e9)
        else:
            mask = torch.ones_like(scores, dtype=torch.float32)
            if dropout is not None:
                mask = dropout(mask)
            mask = mask == 0
            scores = scores.masked_fill(mask, -1e9)
        #print(mask.shape)
        '''
        if self.top_k!=0:
            top_k_values, top_k_indices = torch.topk(scores, k=self.top_k, dim=-1)
            top_k_mask = torch.stack([torch.stack([score_ij.lt(top_k_ij[:, -1])
                                                   for score_ij, top_k_ij in zip(score_i, top_k_i)], dim=0)
                                      for score_i, top_k_i in zip(scores, top_k_values)], dim=0)

            scores = scores.masked_fill(top_k_mask, -1e9)
        '''
        p_attn = self.activation(scores)


        if dropout is not None:
            p_attn = dropout(p_attn)
        #print(p_attn.shape, value.shape)
        return torch.matmul(p_attn, value), p_attn