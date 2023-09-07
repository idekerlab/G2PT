import torch.nn as nn
import torch.nn.functional as F
import torch



class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, activation='softmax', top_k=0, transform=True, n_type=1):
        super().__init__()
        assert d_model % h == 0

        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.n_type = n_type
        if n_type == 1:
            self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for _ in range(3)])
        else:
            self.query_linear = nn.Linear(d_model, d_model, bias=False)
            self.key_linears = nn.ModuleList([nn.Linear(d_model, d_model, bias=False) for i in range(self.n_type)])
            self.value_linear = nn.Linear(d_model, d_model, bias=False)
        self.output_linear = nn.Linear(d_model, d_model, bias=False)
        self.attention = Attention(n_heads=h, activation=activation, top_k=top_k)
        self.transform = transform

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        if self.transform:
            if self.n_type==1:
                query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                                     for l, x in zip(self.linear_layers, (query, key, value))]
            else:
                query = self.query_linear(query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                key = [key_linear(key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) for key_linear in self.key_linears]
                value = self.value_linear(value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.dropout(self.output_linear(self.dropout(x)))

    def get_attention(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return attn

    def get_score(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linear_layers, (query, key, value))]
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=self.dropout)
        return score

class Attention(nn.Module):
    """
    Compute 'Scaled Dot Product Attention
    """

    def __init__(self, n_heads=1, activation='softmax', top_k=0):
        self.n_heads = n_heads
        super().__init__()
        self.activation_type = activation
        if activation=='softmax':
            self.activation = nn.Softmax(dim=-1)
        elif activation=='sigmoid':
            self.activation = nn.Sigmoid()
        elif activation=='tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = None
        #self.top_k = top_k

    def forward(self, query, key, value, mask=None, dropout=None):
        if type(key)==list:
            scores = [torch.matmul(query, key_i.transpose(-2, -1)) \
                     / torch.sqrt(torch.tensor(query.size(-1)))
                      for key_i in key]
            if mask is not None:
                score_sum = []
                for score, mask_i in zip(scores, mask):
                    weight, mask_mat = mask_i
                    mask_to_fill = mask_mat == 0
                    mask_to_weight = mask_mat != 0
                    mask_to_fill = mask_to_fill.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
                    mask_to_weight = mask_to_weight.unsqueeze(1).expand(-1, self.n_heads, -1, -1) * weight
                    if (self.activation_type=='softmax') or (self.activation_type=='sigmoid'):
                        score_sum.append((score*mask_to_weight).masked_fill(mask_to_fill, -1e9))
                    else:
                        score_sum.append((score*mask_to_weight).masked_fill(mask_to_fill, 0))
                scores = sum(score_sum)
            else:
                scores = sum(scores)
        else:
            scores = torch.matmul(query, key.transpose(-2, -1)) \
                     / torch.sqrt(torch.tensor(query.size(-1)))
            if mask is not None:
                mask = mask == 0
                mask = mask.unsqueeze(1).expand(-1, self.n_heads, - 1, -1)
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
        if self.activation:
            p_attn = self.activation(scores)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), p_attn, scores
