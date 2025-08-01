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
        """
        Multi-head Differential Attention.
        """
        super().__init__()
        self.embed_dim = d_model
        self.num_heads = h
        
        # For Grouped-Query Attention
        if h > 1 and h % 2 == 0:
            self.num_kv_heads = h // 2
        else:
            self.num_kv_heads = h
        
        self.n_rep = self.num_heads // self.num_kv_heads
        self.head_dim = d_model // self.num_heads
        self.scaling = 1.0 # As per user feedback and to avoid flat softmax

        # We project to 2x the dimension for queries and keys to get two sets
        self.q_proj = nn.Linear(d_model, 2 * self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, 2 * self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.num_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        self.lambda_init = lambda_init_fn(depth)
        self.lambda_ = nn.Parameter(torch.tensor(self.lambda_init))
        
        self.subln = RMSNorm(d_model, eps=1e-5)

    def get_attention(self, q, k, v, mask=None):
        bsz, tgt_len, _ = q.size()
        bsz, src_len, _ = k.size()

        # Project and split queries and keys for the two attention maps
        q_proj_out = self.q_proj(q)
        k_proj_out = self.k_proj(k)

        q1, q2 = q_proj_out.chunk(2, dim=-1)
        k1, k2 = k_proj_out.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q1 = q1.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k1 = k1.view(bsz, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        q2 = q2.view(bsz, tgt_len, self.num_heads, self.head_dim).transpose(1, 2)
        k2 = k2.view(bsz, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # Project values
        v = self.v_proj(v).view(bsz, src_len, self.num_kv_heads, self.head_dim).transpose(1, 2)

        # Repeat KVs for Grouped-Query Attention
        k1 = repeat_kv(k1, self.n_rep)
        k2 = repeat_kv(k2, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        # Compute attention scores
        attn_scores1 = torch.matmul(q1, k1.transpose(-1, -2)) * self.scaling
        attn_scores2 = torch.matmul(q2, k2.transpose(-1, -2)) * self.scaling

        if mask is not None:
            attn_scores1 = attn_scores1 + mask
            attn_scores2 = attn_scores2 + mask

        # Compute softmax for both attention maps
        attn_weights1 = F.softmax(attn_scores1, dim=-1, dtype=torch.float32).type_as(q)
        attn_weights2 = F.softmax(attn_scores2, dim=-1, dtype=torch.float32).type_as(q)

        # Differential attention
        attn_weights = attn_weights1 - self.lambda_ * attn_weights2
        
        return attn_weights, v

    def forward(self, q, k, v, mask=None):
        bsz, tgt_len, _ = q.size()

        attn_weights, v = self.get_attention(q, k, v, mask=mask)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(bsz, tgt_len, self.embed_dim)
        
        # Gating and LayerNorm
        attn_output = self.subln(attn_output)
        attn_output = attn_output * (1 - self.lambda_init) # Gating with initial lambda

        attn_output = self.out_proj(attn_output)
        return attn_output
