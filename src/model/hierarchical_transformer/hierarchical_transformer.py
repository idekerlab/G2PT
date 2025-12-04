import torch.nn as nn
import torch.nn.functional as F
import torch
import xformers.ops as xops
import math


class SwiGLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int, p: float):  # d_ff keeps API stable
        super().__init__()
        # We still accept d_ff but compute expansion ratio from it.
        expansion = (d_ff // d_model)
        inner = expansion * d_model
        self.w12 = nn.Linear(d_model, inner * 2, bias=False)  # gate + value
        self.proj = nn.Linear(inner, d_model, bias=False)
        self.drop = nn.Dropout(p)

    def forward(self, x: torch.Tensor):
        u, v = self.w12(x).chunk(2, dim=-1)
        return self.proj(self.drop(F.silu(u) * v))


# ---------------------------------------------------------------------------
# 1‑block update  (keeps original signature)
# ---------------------------------------------------------------------------
class HierarchicalTransformerUpdate(nn.Module):
    def __init__(
        self,
        hidden: int,
        attn_heads: int,
        feed_forward_hidden: int,
        norm: nn.Module,
        dropout: float = 0.2,
        attention_dropout: float = 0.1,
        norm_channel_first: bool = False,
        softmax=True,
    ) -> None:
        super().__init__()
        assert hidden % attn_heads == 0, "hidden must divide heads"
        self.h = attn_heads
        self.d_h = hidden // attn_heads
        self.norm_channel_first = norm_channel_first

        # Independent projections for true cross‑attention
        self.q_proj = nn.Linear(hidden, hidden, bias=False)
        self.k_proj = nn.Linear(hidden, hidden, bias=False)
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

        # Norms & FFN
        self.norm_attn = norm                # pre‑supplied
        self.norm_ffn  = nn.LayerNorm(hidden, eps=1e-5)
        self.ffn = SwiGLUFFN(hidden, feed_forward_hidden, dropout)
        self.drop = nn.Dropout(dropout)
        self.attention_dropout = attention_dropout
        self.drop_opposite = nn.Dropout(1-dropout)
        self.softmax = softmax
        #self.gate_attn = nn.Parameter(torch.zeros(1))
        #self.gate_ffn = nn.Parameter(torch.zeros(1))

    def _xops_softmax_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask,
        return_attention=False
    ):
        B, Lq, H, dh = q.shape

        bias = None
        if mask is not None:
            if isinstance(mask, torch.Tensor):
                mask = mask.unsqueeze(1).repeat(1, self.h, 1, 1)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=mask,
            p=self.attention_dropout if self.training else 0.0,
        )
        return out.reshape(B, Lq, H, dh).reshape(B, Lq, H * dh)

    # ---------------------------------------------------------------------
    # unified wrapper  (replaces old _flash_xattn)
    # ---------------------------------------------------------------------
    def _xattn(self, q, k, v, mask):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.h, self.d_h)
        k = self.k_proj(k).view(B, -1, self.h, self.d_h)
        v = self.v_proj(v).view(B, -1, self.h, self.d_h)


        out = self._xops_softmax_attn(q, k, v, mask)
        return self.o_proj(out)

    

    # ---------------------------------------------------------------------
    # forward is unchanged except the private call name
    # ---------------------------------------------------------------------
    def get_attention_scores(self, q, k, mask=None, attention=True):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.h, self.d_h)
        k = self.k_proj(k).view(B, -1, self.h, self.d_h)
        q = q.permute(0, 2, 1, 3)  # (B, H, Lq, d)
        k = k.permute(0, 2, 1, 3)  # (B, H, Lk, d)

        scale_factor = 1.0 / math.sqrt(self.d_h)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

        if mask is not None:
            if mask.dtype == torch.bool:
                scores = scores.masked_fill(mask, float('-inf'))
            else:
                scores = scores + mask

        if attention:
            scores = F.softmax(scores, dim=-1)

        return scores

    def forward(self, q, k, v, mask):
        q_norm = self.norm_attn(q)
        attn_output = self.drop(self._xattn(q_norm, k, v, mask))
        x = q + attn_output
        #gate_a = torch.sigmoid(self.gate_attn)
        #x = q * (1 - gate_a) + attn_out * gate_a

        y = self.norm_ffn(x)
        ffn_out = self.drop(self.ffn(y))
        #gate_f = torch.sigmoid(self.gate_ffn)
        #x = x * (1 - gate_f) + ffn_out * gate_f
        x = x + ffn_out

        if self.norm_channel_first:
            x = x.transpose(-1, -2)
            x = self.norm_attn(x)
            x = x.transpose(-1, -2)
        return x



class PositionWiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.w_2(self.dropout(self.activation(self.w_1(x))))

class HierarchicalTransformer(nn.Module):
    def __init__(self, hidden: object, attn_heads: object, feed_forward_hidden: object, inner_norm: object, outer_norm: object, dropout: object = 0.2,
                 attention_dropout: float = 0.1,
                 conv_type: object = 'system',
                 norm_channel_first: object = False, transform: object = True, n_type: object = 1, activation: object = 'softmax', poincare: object = False) -> object:
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformer, self).__init__()
        self.hierarchical_transformer_update = HierarchicalTransformerUpdate(hidden, attn_heads, feed_forward_hidden, inner_norm,
                                                                     dropout, attention_dropout=attention_dropout, norm_channel_first=norm_channel_first, softmax=activation=='softmax')
        self.norm = outer_norm
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type


    def forward(self, q, k, mask, dropout=True):
        batch_size = q.size(0)
        if (self.conv_type=='system') & (mask is not None):
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1,)
        result = self.hierarchical_transformer_update(q, k, k, mask)

        #result_layer_norm = self.layer_norm(result)
        #if self.norm is not None:
        if self.norm_channel_first:
            result = result.transpose(-1, -2)
            result = self.norm(result)
            result = result.transpose(-1, -2)
                #result = (result + result_layer_norm)/2
        else:
            result = self.norm(result)
                #result = (result + result_layer_norm)/2
        #updated_value = updated_value.permute(0, 2, 1)
        if mask is not None:
            if self.n_type > 1:
                mask = sum([m[1] for m in mask])
            node_mask = torch.sum(mask, dim=-1) == 0
            node_mask = node_mask.unsqueeze(-1).expand(-1, -1, q.size(-1))
            result = result.masked_fill(node_mask, 0)
        return result

    def get_attention(self, q, k, mask):
        batch_size = q.size(0)
        if (self.conv_type == 'system') & (mask is not None):
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.hierarchical_transformer_update.get_attention_scores(q, k, mask=mask)
    def get_score(self, q, k, mask):
        batch_size = q.size(0)
        if (self.conv_type == 'system') & (mask is not None):
            mask = mask.unsqueeze(0).unsqueeze(0).expand(batch_size, -1, -1, -1)
        return self.hierarchical_transformer_update.get_attention_scores(q, k, mask=mask)

