import torch.nn.functional as F
import torch
import math
from torch import nn
from src.model.utils import RMSNorm, euclidian_to_poincare, feature_clipping

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
        if mask is not None:
            attn_weights = attn_weights / mask
        v = self.v_proj(v)
        v = v.view(bsz, src_len, self.num_kv_heads, self.embed_dim)
        v = v.transpose(1, 2)
        attn = torch.matmul(attn_weights, v)
        attn = self.subln(attn)
        attn = attn * (1 - self.lambda_init)
        attn = attn.transpose(1, 2).reshape(bsz, tgt_len, self.num_heads * 2 * self.head_dim)

        attn = self.out_proj(attn)
        return attn
