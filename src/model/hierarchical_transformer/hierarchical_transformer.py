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
        self.softmax = softmax

    def _xops_softmax_attn(
        self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask
    ):
        B, Lq, H, dh = q.shape
        #print(q.size(), k.size(), mask.size())
        #qh = q.reshape(B * H, Lq, dh)
        #kh = k.reshape(B * H, k.size(2), dh)
        #vh = v.reshape(B * H, v.size(2), dh)

        bias = None
        if mask is not None:                            # (B,1|H,Lq,Lk)
            mask = mask.unsqueeze(1).repeat(1, 4, 1, 1)#mask.expand(B, H, -1, -1).reshape(B * H, Lq, -1)
            #bias = xops.fmha.attn_bias.PreconvertedBias(mask, True)

        out = xops.memory_efficient_attention(
            q, k, v,
            attn_bias=mask,
            p=self.drop.p if self.training else 0.0,
        )
        #print(out.shape)# (B, lq, H,dh)
        return out.reshape(B, Lq, H, dh)\
                  .reshape(B, Lq, H * dh)

    # ---------------------------------------------------------------------
    # no-softmax path  (keeps behaviour of your original helper)
    # ---------------------------------------------------------------------
    @staticmethod
    def _dotprod_no_softmax(
        q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
        mask: torch.Tensor, p_drop: float
    ):
        """
        q,k,v : (B, L, H, d)   already split into heads
        mask  : (B, H, Lq, Lk) bool or additive-bias, or None
        returns : (B, Lq, H*d)
        """
        B, Lq, H, d = q.shape
        _, Lk, _, _ = k.shape
        q = q.permute(0, 2, 1, 3)  # (B, H, Lq, d)
        k = k.permute(0, 2, 1, 3)  # (B, H, Lk, d)
        v = v.permute(0, 2, 1, 3)  # (B, H, Lk, d)

        # 1. Scaled Dot-Product Scores
        # (B, H, Lq, d) @ (B, H, d, Lk) -> (B, H, Lq, Lk)
        scale_factor = 1.0 / math.sqrt(d)
        # Using torch.rsqrt can sometimes be slightly more optimized if d is large
        # scale_factor = torch.rsqrt(torch.tensor(d, dtype=q.dtype, device=q.device))
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale_factor

        # 2. Apply Mask (if provided) directly to the scores
        if mask is not None:
            # Ensure mask shape is compatible, potentially requires broadcasting
            # Example: mask shape (B, 1, Lq, Lk) could broadcast over H heads
            if mask.dtype == torch.bool:
                # If True means MASK OUT, set score to -infinity
                # Use masked_fill_ for in-place operation if scores tensor allows
                scores = scores.masked_fill(mask, float('-inf'))
                # If True means KEEP (less common mask convention)
                # scores = scores.masked_fill(~mask, float('-inf'))
            else:
                # Additive mask (assumes 0 where not masked, large negative where masked)
                # Ensure mask is on the same device and dtype or handle conversion
                scores = scores + mask  # Broadcasting might apply here

        # 3. Apply the (potentially masked) scores directly to Values
        #    NO SOFTMAX applied here.
        # (B, H, Lq, Lk) @ (B, H, Lk, d) -> (B, H, Lq, d)
        output = torch.matmul(scores, v)

        # Handle potential -inf * 0 = NaN cases resulting from masking if needed
        # If masked scores are -inf, and corresponding v is 0, matmul result can be NaN.
        # If this is undesirable, you might zero out NaNs, though often NaN propagation
        # is acceptable or indicates an issue elsewhere.
        # output = torch.nan_to_num(output, nan=0.0) # Optional: Replace NaN with 0

        # 4. Combine Heads: Reshape/Permute back to (B, Lq, H*d)
        # (B, H, Lq, d) -> (B, Lq, H, d)
        output = output.permute(0, 2, 1, 3)
        # Ensure contiguous memory layout after permute for reliable view/reshape
        output = output.contiguous()
        # (B, Lq, H, d) -> (B, Lq, H*d)
        output = output.view(B, Lq, H * d)
        # Alternatively: output = output.reshape(B, Lq, -1) # -1 infers H*d

        return output
    # ---------------------------------------------------------------------
    # unified wrapper  (replaces old _flash_xattn)
    # ---------------------------------------------------------------------
    def _xattn(self, q, k, v, mask):
        B = q.size(0)
        q = self.q_proj(q).view(B, -1, self.h, self.d_h)#.transpose(1, 2)
        k = self.k_proj(k).view(B, -1, self.h, self.d_h)#.transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.h, self.d_h)#.transpose(1, 2)

        if self.softmax:
            out = self._xops_softmax_attn(q, k, v, mask)
        else:
            out = self._dotprod_no_softmax(q, k, v, mask, self.drop.p if self.training else 0.0)

        return self.o_proj(out)

    # ---------------------------------------------------------------------
    # forward is unchanged except the private call name
    # ---------------------------------------------------------------------
    def forward(self, q, k, v, mask):
        q_norm = self.norm_attn(q)
        x = q + self.drop(self._xattn(q_norm, k, v, mask))

        y = self.norm_ffn(x)
        x = x + self.drop(self.ffn(y))

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
    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2, conv_type='system',
                 norm_channel_first=False, transform=True, n_type=1, activation='softmax', poincare=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformer, self).__init__()
        self.hierarchical_transformer_update = HierarchicalTransformerUpdate(hidden, attn_heads, feed_forward_hidden, inner_norm,
                                                                     dropout, norm_channel_first=norm_channel_first, softmax=activation=='softmax')
        self.norm = outer_norm
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type


    def forward(self, q, k, mask, dropout=True):
        batch_size = q.size(0)
        if self.conv_type=='system':
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
        if self.conv_type == 'system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1, )
        return self.hierarchical_transformer_update.attention.get_attention(q, k, k, mask=mask)
    def get_score(self, q, k, mask):
        batch_size = q.size(0)
        if self.conv_type == 'system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1, )
        return self.hierarchical_transformer_update.attention.get_score(q, k, k, mask=mask)

'''
class HierarchicalTransformerUpdate(nn.Module):
    """
    Bidirectional Encoder = Transformer (self-attention)
    Transformer = MultiHead_Attention + Feed_Forward with sublayer connection
    """

    def __init__(self, hidden, attn_heads, feed_forward_hidden, norm, dropout=0.2, norm_channel_first=False,
                 n_type=1, transform=True, activation='softmax', poincare=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformerUpdate, self).__init__()
        self.attn_heads = attn_heads
        self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout, transform=transform,
                                              n_type=n_type, activation=activation, poincare=poincare)
        #self.layer_norm = nn.LayerNorm(hidden)
        self.norm = norm
        self.feed_forward = PositionWiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type

    def forward(self, q, k, v, mask=None, dropout=True):
        result = self.attention.forward(q, k, v, mask=mask, dropout=dropout)
        #result_layer_norm = self.layer_norm(result)
        if self.norm_channel_first:
            result = result.transpose(-1, -2)
            result = self.norm(result)
            result = result.transpose(-1, -2)
            #result = (result + result_layer_norm)/2
        else:
            result = self.norm(result)
            #result = (result + result_layer_norm)/2
        result = self.feed_forward(result)
        #result = self.norm(result)
        return self.dropout(result)


class HierarchicalTransformer(nn.Module):
    def __init__(self, hidden, attn_heads, feed_forward_hidden, inner_norm, outer_norm, dropout=0.2, conv_type='system',
                 norm_channel_first=False, transform=True, n_type=1, activation='softmax', poincare=False):
        """
        :param hidden: hidden size of transformer
        :param attn_heads: head sizes of multi-head attention
        :param feed_forward_hidden: feed_forward_hidden, usually 4*hidden_size
        :param norm: normalization layer
        :param dropout: dropout rate
        """

        super(HierarchicalTransformer, self).__init__()
        self.hierarchical_transformer_update = HierarchicalTransformerUpdate(hidden, attn_heads, feed_forward_hidden, inner_norm,
                                                                     dropout, norm_channel_first=norm_channel_first, transform=transform,
                                                                             n_type=n_type, activation=activation, poincare=poincare)
        self.norm = outer_norm
        self.conv_type = conv_type
        self.dropout = nn.Dropout(dropout)
        self.norm_channel_first = norm_channel_first
        self.n_type = n_type


    def forward(self, q, k, mask, dropout=True):
        batch_size = q.size(0)
        if self.conv_type=='system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1,)
        result = self.hierarchical_transformer_update(q, k, k, mask, dropout=dropout)

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
        if self.conv_type == 'system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1, )
        return self.hierarchical_transformer_update.attention.get_attention(q, k, k, mask=mask)
    def get_score(self, q, k, mask):
        batch_size = q.size(0)
        if self.conv_type == 'system':
            mask = mask.unsqueeze(0).expand(batch_size, -1, -1, )
        return self.hierarchical_transformer_update.attention.get_score(q, k, k, mask=mask)
'''