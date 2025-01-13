import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from torch import nn
from src.model.utils import RMSNorm, euclidian_to_poincare, feature_clipping


class MultiHeadedAttention(nn.Module):
    """
    Take in model size and number of heads.
    """

    def __init__(self, h, d_model, dropout=0.1, activation='softmax', transform=True, n_type=1,
                 poincare=False):
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
        self.attention = Attention(n_heads=h, activation=activation, poincare=poincare)
        self.transform = transform
        self.poincare = poincare
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, dropout=True):
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
        if dropout:
            dropout = self.dropout
        else:
            dropout = None
        x, attn, score = self.attention(query, key, value, mask=mask, dropout=dropout)

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

    def __init__(self, n_heads=1, activation='softmax', poincare=False):
        super().__init__()
        self.n_heads = n_heads
        self.poincare = poincare

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

    def poincare_distance(self, x, y, eps=1e-5):
        """
        Compute pairwise Poincaré distances between x and y.
        x: (batch_size, seq_len, d_model)
        y: (batch_size, seq_len, d_model)  or  (batch_size, seq_len, d_model)

        We'll broadcast them to shape (batch_size, seq_len, seq_len, d_model)
        and compute distance for each pair.
        """
        # x, y => (batch_size, seq_len, 1, d_model), (batch_size, 1, seq_len, d_model)
        # so we can broadcast differences across pairs
        x_expanded = x.unsqueeze(3)  # (B, L, 1, d_model)
        y_expanded = y.unsqueeze(2)  # (B, 1, L, d_model)


        # Norm-squared
        x_norm_sq = torch.sum(x_expanded ** 2, dim=-1, keepdim=True)  # (B, L, 1, 1)
        y_norm_sq = torch.sum(y_expanded ** 2, dim=-1, keepdim=True)  # (B, 1, L, 1)
        # Euclidean difference
        diff = x_expanded - y_expanded  # (B, L, L, d_model)
        diff_norm_sq = torch.sum(diff ** 2, dim=-1)  # (B, L, L)

        # Poincaré distance formula:
        # d(x,y) = arcosh( 1 + 2 ||x - y||^2 / ((1 - ||x||^2)(1 - ||y||^2)) )
        # clamp denominators to avoid zero division
        denom = (1.0 - x_norm_sq) * (1.0 - y_norm_sq)  # (B, L, L, 1)
        denom = torch.clamp(denom, min=eps)

        arg = 1.0 + 2.0 * diff_norm_sq / denom.squeeze(-1)  # (B, L, L)
        arg = torch.clamp(arg, min=1.0 + eps)  # ensure >= 1 for arcosh

        return torch.acosh(arg)

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
            if self.poincare:
                query = euclidian_to_poincare(feature_clipping(query))
                key = euclidian_to_poincare(feature_clipping(key))
                scores = - self.poincare_distance(query, key)
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


class FewShotAttention(Attention):

    def __init__(self, d_model, n_heads=1, n_train_celllines=1, activation='softmax', dropout=0.2):
        super(FewShotAttention, self).__init__(n_heads=n_heads, activation=activation)
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.cellline_weights = nn.Parameter(torch.ones(n_train_celllines), requires_grad=True)
        #self.query_norm = nn.LayerNorm(d_model)
        #self.key_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None, dropout=None, transform=True):
        batch_size = query.size(0)
        #query = self.query_norm(query)
        #key = self.key_norm(key)
        query = query.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        key = key.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) \
                 / torch.sqrt(torch.tensor(query.size(-1)))
        if transform:
            scores = scores * self.cellline_weights.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1)
        mask = torch.ones_like(scores)
        mask = self.dropout(mask)
        scores = scores.masked_fill(mask==0, -1e9)
        p_attn = self.activation(scores)

        x = torch.matmul(p_attn, value)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return x, p_attn, scores


## Differential Transformer


def init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
            .expand(bs, n_kv_heads, n_rep, slen, head_dim)
            .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )


def lambda_init_fn(depth):
    return 0.8 - 0.6 * math.exp(-0.3 * depth)


class MultiheadDiffAttn(nn.Module):
    def __init__(self, h, d_model, depth):
        '''
        Multi-head Diff Attn for special case with one-head
        '''
        super().__init__()
        self.embed_dim = d_model
        # num_heads set to half of Transformer's #heads
        self.num_heads = h #// args.model_parallel_size
        self.num_kv_heads = h #// 2
        self.n_rep = 2#self.num_heads // self.num_kv_heads

        self.head_dim = d_model // 2
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model // self.n_rep, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_q1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k1 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_q2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))
        self.lambda_k2 = nn.Parameter(torch.zeros(self.head_dim, dtype=torch.float32).normal_(mean=0, std=0.1))

        self.subln = RMSNorm(2 * self.head_dim, eps=1e-5, elementwise_affine=False)

    def get_attention(
            self,
            q, k, v,
            mask=None,
    ):
        bsz, tgt_len, embed_dim = q.size()
        bsz, src_len, embed_dim = k.size()

        q = self.q_proj(q)
        k = self.k_proj(k)

        q = q.view(bsz, tgt_len, 2 * self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_kv_heads, self.head_dim)

        offset = src_len - tgt_len
        q = q.transpose(1, 2)
        k = repeat_kv(k.transpose(1, 2), self.n_rep)
        #v = repeat_kv(v.transpose(1, 2), self.n_rep)
        q *= self.scaling

        attn_weights = torch.matmul(q, k.transpose(-1, -2))
        '''
        if mask is None:
            mask = torch.triu(
                torch.zeros([tgt_len, src_len])
                    .float()
                    .fill_(float("-inf"))
                    .type_as(attn_weights),
                    1 + offset,
            )
        attn_weights = torch.nan_to_num(attn_weights)
        attn_weights += mask
        '''
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
            attn_weights
        )
        lambda_1 = torch.exp(torch.sum(self.lambda_q1 * self.lambda_k1, dim=-1).float()).type_as(q)
        lambda_2 = torch.exp(torch.sum(self.lambda_q2 * self.lambda_k2, dim=-1).float()).type_as(q)
        lambda_full = lambda_1 - lambda_2 + self.lambda_init
        attn_weights = attn_weights.view(bsz, self.num_heads, 2, tgt_len, src_len)
        attn_weights = attn_weights[:, :, 0] - lambda_full * attn_weights[:, :, 1]
        return attn_weights


    def forward(self, q, k, v, mask=None):

        bsz, tgt_len, embed_dim = q.size()
        bsz, src_len, embed_dim = k.size()

        attn_weights = self.get_attention(q, k, v, mask=mask)
        v = self.v_proj(v)
        v = v.view(bsz, src_len, self.num_kv_heads, self.embed_dim)
        v = v.transpose(1, 2)
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn